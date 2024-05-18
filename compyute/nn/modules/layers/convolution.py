"""Convolution layers module"""

from typing import Literal, Optional

from ....random import uniform
from ....tensor_functions import zeros
from ....tensors import Tensor
from ....types import DtypeLike
from ...functional import avgpooling2d, convolve1d, convolve2d, maxpooling2d
from ...parameter import Parameter
from ..module import Module

__all__ = ["Convolution1d", "Convolution2d", "MaxPooling2d", "AvgPooling2d"]


class Convolution1d(Module):
    """Layer used for spacial information and feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Literal["causal", "same", "valid"] = "causal",
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        dtype: DtypeLike = "float32",
        label: Optional[str] = None,
    ) -> None:
        """Convolutional layer used for temporal information and feature extraction.
        Input: (B, Ci, Ti)
            B ... batch, Ci ... input channels, Ti ... input time
        Output: (B, Co, To)
            B ... batch, Co ... output channels, To ... output time

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels (filters).
        kernel_size : int
            Size of each kernel.
        padding: Literal["causal", "same", "valid"], optional
            Padding applied to a tensor before the convolution, by default "causal".
        stride : int, optional
            Stride used for the convolution operation, by default 1.
        dilation : int, optional
            Dilation used for each axis of the filter, by default 1.
        bias : bool, optional
            Whether to use bias values, by default True.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        label: str, optional
            Module label.
        """
        super().__init__(label)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.dtype = dtype

        # init weights
        # (Co, Ci, K)
        k = (in_channels * kernel_size) ** -0.5
        w = uniform((out_channels, in_channels, kernel_size), -k, k, dtype=dtype)
        self.w = Parameter(w, label="w")

        # init biases
        # (Co,)
        self.b = None
        if bias:
            b = zeros((out_channels,), dtype=dtype)
            self.b = Parameter(b, label="b")

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [3])
        x = x.astype(self.dtype)

        # flip weights for cross correlation
        w_flip = self.w.flip(-1)

        x = x.insert_dim(axis=1)  # (B, 1, Ci, Ti)
        w_flip = w_flip.reshape((1, *w_flip.shape))  # (1, Co, Ci, K)

        # convolve
        # (B, 1, Ci, Ti) * (1, Co, Ci, K) -> (B, Co, Ci, To)
        x_conv_w = convolve1d(x, w_flip, self.padding, self.stride, self.dilation)

        # sum over input channels
        # (B, Co, Ci, To) -> (B, Co, To)
        y = x_conv_w.sum(axis=2)

        if self.b is not None:
            # (B, Co, To) + (Co, 1)
            y += self.b.reshape((*self.b.shape, 1))

        if self.training:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.astype(self.dtype)

                Ti = x.shape[-1]
                B, Co, To = dy.shape
                K = self.kernel_size
                S = self.stride
                D = self.dilation

                # fill elements skipped by strides with zeros
                dy_p = zeros((B, Co, S * To), dtype=self.dtype, device=self.device)
                dy_p[:, :, ::S] = dy
                dy_p_ti = 1 + (Ti - K) if self.padding == "valid" else Ti
                dy_p = dy_p[:, :, :dy_p_ti]

                # ----------------
                # input grads
                # ----------------
                dy_p_ext = dy_p.insert_dim(axis=2)  # (B, Co, 1, To)
                w_ext = self.w.reshape((1, *self.w.shape))  # (1, Co, Ci, K)
                padding = "full" if self.padding == "valid" else self.padding

                # convolve
                # (B, Co, 1, To) * (1, Co, Ci, K)
                dy_conv_w = convolve1d(dy_p_ext, w_ext, padding, 1, D)

                # sum over output channels
                # (B, Ci, Ti)
                dx = dy_conv_w.sum(axis=1)

                # ----------------
                # weight grads
                # ----------------
                if self.w.requires_grad:
                    dy_p_ext = dy_p_ext.flip(-1)

                    # convolve
                    # (B, 1, Ci, Ti) * (B, Co, 1, To) -> (B, Co, Ci, K)
                    x_conv_dy = convolve1d(x, dy_p_ext, padding)
                    if self.padding == "causal":
                        x_conv_dy = x_conv_dy[:, :, :, -K * D :: D]
                    else:
                        k_size = (K - 1) * D + 1
                        k = (x_conv_dy.shape[-1] - k_size) // 2
                        x_conv_dy = x_conv_dy[:, :, :, k : k + k_size : D]

                    # sum over batches
                    # (B, Co, Ci, K) -> (Co, Ci, K)
                    self.w.grad = x_conv_dy.sum(axis=0)

                # ----------------
                # bias grads
                # ----------------
                if self.b is not None and self.b.requires_grad:
                    # sum over batches and time
                    # (B, Co, To) -> (Co,)
                    self.b.grad = dy.sum(axis=(0, 2))

                return dx

            self._backward = _backward

        return y


class Convolution2d(Module):
    """Layer used for spacial information and feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: Literal["same", "valid"] = "valid",
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        dtype: DtypeLike = "float32",
        label: Optional[str] = None,
    ) -> None:
        """Convolutional layer used for spacial information and feature extraction.
        Input: (B, Ci, Yi, Xi)
            B ... batch, Ci ... input channels, Yi ... input height, Xi ... input width
        Output: (B, Co, Yo, Xo)
            B ... batch, Co ... output channels, Yo ... output height, Xo ... output width

        Parameters
        ----------
        in_channels : int
            Number of input channels (color channels).
        out_channels : int
            Number of output channels (filters).
        kernel_size : int, optional
            Size of each kernel, by default 3.
        padding: Literal["same", "valid"], optional
            Padding applied to a tensor before the convolution, by default "valid".
        stride : int , optional
            Strides used for the convolution operation, by default 1.
        dilation : int , optional
            Dilations used for each axis of the filter, by default 1.
        bias : bool, optional
            Whether to use bias values, by default True.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        label: str, optional
            Module label.
        """
        super().__init__(label)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.dtype = dtype

        # init weights
        # (Co, Ci, Ky, Kx)
        k = (in_channels * self.kernel_size**2) ** -0.5
        w = uniform((out_channels, in_channels, self.kernel_size, self.kernel_size), -k, k, dtype)
        self.w = Parameter(w, label="w")

        # init biases
        # (Co,)
        self.b = Parameter(zeros((out_channels,), dtype), label="b") if bias else None

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [4])
        x = x.astype(self.dtype)

        # rotate weights for cross correlation
        w_flip = self.w.flip((-2, -1))

        x = x.insert_dim(axis=1)  # (B, 1, Ci, Yi, Xi)
        w_flip = w_flip.reshape((1, *w_flip.shape))  # (1, Co, Ci, K, K)

        # convolve
        # (B, 1, Ci, Yi, Xi) * (1, Co, Ci, K, K) -> (B, Co, Ci, Yo, Xo)
        x_conv_w = convolve2d(x, w_flip, self.padding, self.stride, self.dilation)

        # sum over input channels
        # (B, Co, Ci, Yo, Xo) -> (B, Co, Yo, Xo)
        y = x_conv_w.sum(axis=2)

        if self.b is not None:
            # (B, Co, Yo, Xo) + (Co, 1, 1)
            y += self.b.add_dims(target_dims=3)

        if self.training:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.astype(self.dtype)

                Yi, Xi = x.shape[-2:]
                B, Co, Yo, Xo = dy.shape
                K = self.kernel_size
                S = self.stride
                D = self.dilation

                # fill elements skipped by strides with zeros
                dy_p = zeros((B, Co, S * Yo, S * Xo), dtype=self.dtype, device=self.device)
                dy_p[:, :, ::S, ::S] = dy
                dy_p_yi = 1 + (Yi - K) if self.padding == "valid" else Yi
                dy_p_xi = 1 + (Xi - K) if self.padding == "valid" else Xi
                dy_p = dy_p[:, :, :dy_p_yi, :dy_p_xi]

                # ----------------
                # input grads
                # ----------------
                dy_p_ext = dy_p.insert_dim(axis=2)  # (B, Co, 1, Yo, Xo)
                w_ext = self.w.reshape((1, *self.w.shape))  # (1, Co, Ci, K, K)
                padding = "full" if self.padding == "valid" else self.padding

                # convolve
                # (B, Co, 1, Yo, Xo) * (1, Co, Ci, K, K) -> (B, Co, Ci, Yi, Xi)
                dy_conv_w = convolve2d(dy_p_ext, w_ext, padding, 1, D)

                # sum over c_out
                # (B, Co, Ci, Yi, Xi) -> (B, Ci, Yi, Xi)
                dx = dy_conv_w.sum(axis=1)

                # ----------------
                # weight grads
                # ----------------
                if self.w.requires_grad:
                    dy_p_ext = dy_p_ext.flip((-2, -1))

                    # convolve
                    # (B, 1, Ci, Yi, Xi) * (B, Co, 1, Yo, Xo) -> (B, Co, Ci, K, K)
                    x_conv_dy = convolve2d(x, dy_p_ext, padding)
                    k_size = (K - 1) * D + 1
                    k = (x_conv_dy.shape[-1] - k_size) // 2
                    x_conv_dy = x_conv_dy[:, :, :, k : k + k_size : D, k : k + k_size : D]

                    # sum over batches
                    # (B, Co, Ci, K, K) -> (Co, Ci, K, K)
                    self.w.grad = x_conv_dy.sum(axis=0)

                # ----------------
                # bias grads
                # ----------------
                if self.b is not None and self.b.requires_grad:
                    # sum over batches, height and width
                    # (B, Co, Yo, Xo) -> (Co,)
                    self.b.grad = dy.sum(axis=(0, 2, 3))

                return dx

            self._backward = _backward

        return y


class MaxPooling2d(Module):
    """MaxPoling layer used to reduce information to avoid overfitting."""

    def __init__(self, kernel_size: int = 2, label: Optional[str] = None) -> None:
        """MaxPoling layer used to reduce information to avoid overfitting.

        Parameters
        ----------
        kernel_size : int, optional
             Shape of the pooling window used for the pooling operation, by default 2.
        label: str, optional
            Module label.
        """
        super().__init__(label)
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [4])

        kernel_size = (self.kernel_size, self.kernel_size)
        y, self._backward = maxpooling2d(x, kernel_size, self.training)
        return y


class AvgPooling2d(Module):
    """AvgPooling layer used to reduce information to avoid overfitting."""

    def __init__(self, kernel_size: int = 2, label: Optional[str] = None) -> None:
        """AvgPooling layer used to reduce information to avoid overfitting.

        Parameters
        ----------
        kernel_size : int, optional
             Shape of the pooling window used for the pooling operation, by default 2.
        label: str, optional
            Module label.
        """
        super().__init__(label)
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [4])

        kernel_size = (self.kernel_size, self.kernel_size)
        y, self._backward = avgpooling2d(x, kernel_size, self.training)
        return y
