"""Convolutional layers module"""

from __future__ import annotations

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ArrayLike
from walnut.nn.funcional import convolve1d, convolve2d, stretch2d
from walnut.nn.module import Module


__all__ = ["Convolution1d", "Convolution2d", "MaxPooling2d"]


class Convolution1d(Module):
    """Layer used for spacial information and feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pad: str = "causal",
        stride: int = 1,
        dil: int = 1,
        weights: Tensor | None = None,
        use_bias: bool = True,
    ) -> None:
        """Convolutional layer used for spacial information and feature extraction.

        Parameters
        ----------
        in_channels : int
            Number of input channels of the layer.
        out_channels : int
            Number of output channels of the layer.
        kernel_size : int
            Shape of each kernel.
        pad: str, optional
            Padding applied before convolution.
            Options are "causal", "valid" or "same", by default "causal".
        stride : int, optional
            Stride used for the convolution operation, by default 1.
        dil : int, optional
            Dilation used for each axis of the filter, by default 1.
        weights : Tensor | None, optional
            Weights of the layer, by default None. If None, weights are initialized randomly.
        use_bias : bool, optional
            Whether to use bias values, by default True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pad = pad
        self.stride = stride
        self.dil = dil
        self.use_bias = use_bias

        # init weights (c_out, c_in, x)
        if weights is None:
            k = int(in_channels * kernel_size) ** -0.5
            self.w = tu.randu((out_channels, in_channels, kernel_size), -k, k)
        else:
            self.w = weights
        self.parameters = [self.w]

        # init bias (c_out,)
        if use_bias:
            self.b = tu.zeros((out_channels,))
            self.parameters += [self.b]

    def __repr__(self) -> str:
        name = self.__class__.__name__
        in_channels = self.in_channels
        out_channels = self.out_channels
        kernel_size = self.kernel_size
        pad = self.pad
        stride = self.stride
        dil = self.dil
        use_bias = self.use_bias
        return f"{name}({in_channels=}, {out_channels=}, {kernel_size=}, {pad=}, {stride=}, {dil=}, {use_bias=})"

    def __call__(self, x: Tensor) -> Tensor:
        # rotate weights for cross correlation
        w_rot = self.w.flip(-1)

        # convolve (b, 1, c_in, x) * (1, c_out, c_in, x)
        x_ext = tu.expand_dims(x, 1)
        w_rot_ext = tu.expand_dims(w_rot, 0)
        x_conv_w = convolve1d(x_ext, w_rot_ext, self.stride, self.dil, self.pad)

        # sum over input channels
        y = x_conv_w.sum(axis=2)

        if self.use_bias:
            y = y + tu.match_dims(self.b, y.ndim - 1)

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)

                x3 = x.shape[-1]
                w3 = self.w.shape[-1]
                dy1, dy2, dy3 = dy.shape

                # undo strides by filling with zeros
                dy_p = tu.zeros((dy1, dy2, self.stride * dy3), device=self.device).data
                dy_p[:, :, :: self.stride] = dy
                out = 1 + (x3 - w3) if self.pad == "valid" else x3
                dy_p = Tensor(dy_p[:, :, :out], device=self.device)

                # input grads (b, c_in, x)
                dy_p_ext = tu.expand_dims(dy_p, 2)
                w_ext = tu.expand_dims(self.w, 0)
                # convolve (b, c_out, 1, x) * (1, c_out, c_in, x)
                pad = (
                    "full"
                    if self.pad == "valid"
                    else "causal"
                    if self.pad == "causal"
                    else "same"
                )
                dy_conv_w = convolve1d(dy_p_ext, w_ext, dil=self.dil, pad=pad)
                # sum over output channels
                dx = dy_conv_w.sum(axis=1).data

                # weight grads (c_out, c_in, x)
                dy_p_ext = dy_p_ext.flip(-1)
                # convolve (b, 1, c_in, x) * (b, c_out, 1, x)
                pad = (
                    w3 // 2 * self.dil
                    if self.pad == "same"
                    else (w3 // 2 * self.dil * 2, 0)
                    if self.pad == "causal"
                    else "valid"
                )
                x_conv_dy = convolve1d(x_ext, dy_p_ext, pad=pad)[
                    :, :, :, -w3 * self.dil :
                ]
                # sum over batches
                self.w.grad = x_conv_dy[:, :, :, :: self.dil].sum(axis=0).data

                # bias grads (c_out,)
                if self.use_bias:
                    self.b.grad = dy.sum(axis=(0, 2))  # sum over b and x

                return dx

            self.backward = backward

        self.set_y(y)
        return y


class Convolution2d(Module):
    """Layer used for spacial information and feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        pad: str = "valid",
        stride: int | tuple[int, int] = 1,
        dil: int | tuple[int, int] = 1,
        weights: Tensor | None = None,
        use_bias: bool = True,
    ) -> None:
        """Convolutional layer used for spacial information and feature extraction.

        Parameters
        ----------
        in_channels : int
            Number of input channels of the layer.
        out_channels : int
            Number of output channels (neurons) of the layer.
        kernel_size : ShapeLike, optional
            Shape of each kernel, by default (3, 3).
        pad: str, optional
            Padding applied before convolution.
            Options are "valid" and "same", by default "valid".
        stride : int | tuple [int, int], optional
            Strides used for the convolution operation, by default 1.
        dil : int | tuple [int, int], optional
            Dilations used for each axis of the filter, by default 1.
        weights : Tensor | None, optional
            Weights of the layer, by default None. If None, weights are initialized randomly.
        use_bias : bool, optional
            Whether to use bias values, by default True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pad = pad
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.dil = (dil, dil) if isinstance(dil, int) else dil
        self.use_bias = use_bias

        # init weights (c_out, c_in, y, x)
        if weights is None:
            k = int(in_channels * tu.prod(kernel_size)) ** -0.5
            self.w = tu.randu((out_channels, in_channels, *kernel_size), -k, k)
        else:
            self.w = weights
        self.parameters = [self.w]

        # init bias (c_out,)
        if self.use_bias:
            self.b = tu.zeros((out_channels,))
            self.parameters += [self.b]

    def __repr__(self) -> str:
        name = self.__class__.__name__
        in_channels = self.in_channels
        out_channels = self.out_channels
        kernel_size = self.kernel_size
        pad = self.pad
        stride = self.stride
        dil = self.dil
        use_bias = self.use_bias
        return f"{name}({in_channels=}, {out_channels=}, {kernel_size=}, {pad=}, {stride=}, {dil=}, {use_bias=})"

    def __call__(self, x: Tensor) -> Tensor:
        # rotate weights for cross correlation
        w_rot = self.w.flip((-2, -1))

        # convolve (b, 1, c_in, y, x) * (1, c_out, c_in, y, x)
        x_ext = tu.expand_dims(x, 1)  # add fake c_out dim
        w_rot_ext = tu.expand_dims(w_rot, 0)  # add fake b dim
        x_conv_w = convolve2d(x_ext, w_rot_ext, self.stride, self.dil, self.pad)

        # sum over input channels
        y = x_conv_w.sum(axis=2)

        if self.use_bias:
            y = y + tu.match_dims(x=self.b, dims=y.ndim - 1)

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)

                w3, w4 = self.w.shape[-2:]
                x3, x4 = x.shape[-2:]
                dy1, dy2, dy3, dy4 = dy.shape
                s1, s2 = self.stride
                d1, d2 = self.dil

                # fill elements skipped by strides with zeros
                dy_p = tu.zeros((dy1, dy2, s1 * dy3, s2 * dy4), device=self.device).data
                dy_p[:, :, ::s1, ::s2] = dy
                out_y = 1 + (x3 - w3) if self.pad == "valid" else x3
                out_x = 1 + (x4 - w4) if self.pad == "valid" else x4
                dy_p = Tensor(dy_p[:, :, :out_y, :out_x], device=self.device)

                # input grads (b, c_in, y, x)
                dy_p_ext = tu.expand_dims(dy_p, 2)
                w_ext = tu.expand_dims(self.w, 0)
                # convolve (b, c_out, 1, y, x) * (1, c_out, c_in, y, x)
                pad = "full" if self.pad == "valid" else "same"
                dy_conv_w = convolve2d(dy_p_ext, w_ext, dil=self.dil, pad=pad)
                dx = dy_conv_w.sum(axis=1).data  # sum over output channels

                # weight grads (c_out, c_in, y, x)
                dy_p_ext = dy_p_ext.flip((-2, -1))
                # convolve (b, 1, c_in, y, x) * (b, c_out, 1, y, x)
                pad = (w3 // 2 * d1, w4 // 2 * d2) if self.pad == "same" else "valid"
                x_conv_dy = convolve2d(x_ext, dy_p_ext, pad=pad)[
                    :, :, :, -w3 * d1 :, -w4 * d2 :
                ]
                # sum over batches
                self.w.grad = x_conv_dy[:, :, :, ::d1, ::d2].sum(axis=0).data

                # bias grads (c_out,)
                if self.use_bias:
                    self.b.grad = dy.sum(axis=(0, 2, 3))  # sum over b, y and x

                return dx

            self.backward = backward

        self.set_y(y)
        return y


class MaxPooling2d(Module):
    """MaxPoling layer used to reduce information to avoid overfitting."""

    def __init__(self, kernel_size: tuple[int, int] = (2, 2)) -> None:
        """MaxPoling layer used to reduce information to avoid overfitting.

        Parameters
        ----------
        kernel_size : tuple[int, int], optional
             Shape of the pooling window used for the pooling operation, by default (2, 2).
        """
        super().__init__()
        self.kernel_size = kernel_size

    def __repr__(self) -> str:
        name = self.__class__.__name__
        kernel_size = self.kernel_size
        return f"{name}({kernel_size=})"

    def __call__(self, x: Tensor) -> Tensor:
        # cut off values to fit the pooling window
        y_fit = x.shape[-2] // self.kernel_size[0] * self.kernel_size[0]
        x_fit = x.shape[-1] // self.kernel_size[1] * self.kernel_size[1]
        x_crop = x[:, :, :y_fit, :x_fit]

        p_y, p_x = self.kernel_size
        x_b, x_c, _, _ = x.shape
        _, _, xc_y, xc_x = x_crop.shape
        y = tu.zeros(
            (x_b, x_c, xc_y // p_y, xc_x // p_x),
            device=self.device,
        )
        for yi in range(y.shape[-2]):
            for xi in range(y.shape[-1]):
                cnk = x.data[:, :, yi * p_y : (yi + 1) * p_y, xi * p_x : (xi + 1) * p_x]
                y[:, :, yi, xi] = cnk.max(axis=(-2, -1))

        y_s = stretch2d(y, self.kernel_size, x_crop.shape)
        p_map = (x_crop == y_s) * 1.0

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)

                dy_s = stretch2d(
                    Tensor(dy, device=self.device), self.kernel_size, p_map.shape
                )
                # use p_map as mask for grads
                return (dy_s * p_map).resize(x.shape).data

            self.backward = backward

        self.set_y(y)
        return y
