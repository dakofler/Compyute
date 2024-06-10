"""Convolution layers module"""

from typing import Literal, Optional

from ...._tensor import Tensor
from ...._tensor_functions._creating import zeros
from ...._types import _DtypeLike
from ....random import uniform
from ...functional import avgpooling2d, convolve1d, convolve2d, maxpooling2d
from ...parameter import Parameter
from .._module import Module

__all__ = ["Convolution1d", "Convolution2d", "MaxPooling2d", "AvgPooling2d"]


class Convolution1d(Module):
    """Layer used for spacial information and feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Literal["valid", "same"] = "valid",
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        dtype: _DtypeLike = "float32",
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
        padding: int, optional
            Padding applied to the input tensor, by default 0.
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
        y, grad_func = convolve1d(
            x, self.w, self.b, self.padding, self.stride, self.dilation, self.training
        )

        if self.training:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.astype(self.dtype)
                dx, dw, db = grad_func(dy)

                if dw is not None:
                    self.w.grad += dw

                if db is not None:
                    self.b.grad += db

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
        dtype: _DtypeLike = "float32",
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
        y, grad_func = convolve2d(
            x, self.w, self.b, self.padding, self.stride, self.dilation, self.training
        )

        if self.training:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.astype(self.dtype)
                dx, dw, db = grad_func(dy)

                if dw is not None:
                    self.w.grad += dw

                if db is not None:
                    self.b.grad += db

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
