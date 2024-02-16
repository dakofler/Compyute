"""Convolutional layers module"""

from __future__ import annotations

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ArrayLike
from walnut.nn.funcional import convolve1d, convolve2d, stretch2d
from walnut.nn.module import Module
from walnut.nn.parameter import Parameter


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
        weights: Parameter | None = None,
        use_bias: bool = True,
        dtype: str = "float32",
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
        weights : Parameter | None, optional
            Weights of the layer, by default None. If None, weights are initialized randomly.
        use_bias : bool, optional
            Whether to use bias values, by default True.
        dtype: str, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pad = pad
        self.stride = stride
        self.dil = dil
        self.use_bias = use_bias
        self.dtype = dtype

        # init weights (c_out, c_in, x)
        if weights is None:
            k = int(in_channels * kernel_size) ** -0.5
            self.w = Parameter(
                tu.randu((out_channels, in_channels, kernel_size), -k, k),
                dtype=dtype,
                label="w",
            )
        else:
            self.w = weights

        # init bias (c_out,)
        if use_bias:
            self.b = Parameter(tu.zeros((out_channels,)), dtype=dtype, label="b")

    def __repr__(self) -> str:
        name = self.__class__.__name__
        in_channels = self.in_channels
        out_channels = self.out_channels
        kernel_size = self.kernel_size
        pad = self.pad
        stride = self.stride
        dil = self.dil
        use_bias = self.use_bias
        dtype = self.dtype
        return f"{name}({in_channels=}, {out_channels=}, {kernel_size=}, {pad=}, {stride=}, {dil=}, {use_bias=}, {dtype=})"

    def __call__(self, x: Tensor) -> Tensor:
        x = x.astype(self.dtype)

        # rotate weights for cross correlation
        w_rot = self.w.flip(-1)

        # convolve (b, 1, c_in, x) * (1, c_out, c_in, x)
        x_ext = tu.expand_dims(x, 1)
        w_rot_ext = tu.expand_dims(w_rot, 0)
        x_conv_w = convolve1d(x_ext, w_rot_ext, self.stride, self.dil, self.pad)
        x_conv_w = x_conv_w.astype(self.dtype)  # conv returns float64

        # sum over input channels
        y = x_conv_w.sum(axis=2)

        if self.use_bias:
            y += tu.match_dims(self.b, y.ndim - 1)

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                dy = dy.astype(self.dtype)
                self.set_dy(dy)

                w_x = self.w.shape[-1]
                x_x = x.shape[-1]
                dy_b, dy_k, dy_x = dy.shape

                # undo strides by filling with zeros
                dy_p_shape = (dy_b, dy_k, self.stride * dy_x)
                dy_p = tu.zeros(dy_p_shape, device=self.device).data
                dy_p[:, :, :: self.stride] = dy
                dy_p_x = 1 + (x_x - w_x) if self.pad == "valid" else x_x
                dy_p = Tensor(dy_p[:, :, :dy_p_x], device=self.device)

                # input grads (b, c_in, x)
                dy_p_ext, w_ext = tu.expand_dims(dy_p, 2), tu.expand_dims(self.w, 0)
                pad = "full" if self.pad == "valid" else self.pad
                # convolve (b, c_out, 1, x) * (1, c_out, c_in, x)
                dy_conv_w = convolve1d(dy_p_ext, w_ext, dil=self.dil, pad=pad)
                dy_conv_w = dy_conv_w.astype(self.dtype)  # conv returns float64
                dx = dy_conv_w.sum(axis=1).data  # sum over output channels

                # weight grads (c_out, c_in, x)
                dy_p_ext = dy_p_ext.flip(-1)

                match self.pad:
                    case "same":
                        pad = w_x // 2 * self.dil
                    case "causal":
                        pad = (w_x // 2 * self.dil * 2, 0)
                    case _:
                        pad = self.pad

                # convolve (b, 1, c_in, x) * (b, c_out, 1, x)
                x_conv_dy = convolve1d(x_ext, dy_p_ext, pad=pad)[
                    :, :, :, -w_x * self.dil :
                ]
                x_conv_dy = x_conv_dy.astype(self.dtype)  # conv returns float64
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
        weights: Parameter | None = None,
        use_bias: bool = True,
        dtype: str = "float32",
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
        weights : Parameter | None, optional
            Weights of the layer, by default None. If None, weights are initialized randomly.
        use_bias : bool, optional
            Whether to use bias values, by default True.
        dtype: str, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pad = pad
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.dil = (dil, dil) if isinstance(dil, int) else dil
        self.use_bias = use_bias
        self.dtype = dtype

        # init weights (c_out, c_in, y, x)
        if weights is None:
            k = int(in_channels * tu.prod(kernel_size)) ** -0.5
            self.w = Parameter(
                tu.randu((out_channels, in_channels, *kernel_size), -k, k),
                dtype=dtype,
                label="w",
            )
        else:
            self.w = weights

        # init bias (c_out,)
        if self.use_bias:
            self.b = Parameter(tu.zeros((out_channels,)), dtype=dtype, label="b")

    def __repr__(self) -> str:
        name = self.__class__.__name__
        in_channels = self.in_channels
        out_channels = self.out_channels
        kernel_size = self.kernel_size
        pad = self.pad
        stride = self.stride
        dil = self.dil
        use_bias = self.use_bias
        dtype = self.dtype
        return f"{name}({in_channels=}, {out_channels=}, {kernel_size=}, {pad=}, {stride=}, {dil=}, {use_bias=}, {dtype=})"

    def __call__(self, x: Tensor) -> Tensor:
        x = x.astype(self.dtype)

        # rotate weights for cross correlation
        w_rot = self.w.flip((-2, -1))

        # convolve (b, 1, c_in, y, x) * (1, c_out, c_in, y, x)
        x_ext, w_rot_ext = tu.expand_dims(x, 1), tu.expand_dims(w_rot, 0)
        x_conv_w = convolve2d(x_ext, w_rot_ext, self.stride, self.dil, self.pad)
        x_conv_w = x_conv_w.astype(self.dtype)  # conv returns float64

        # sum over input channels
        y = x_conv_w.sum(axis=2)

        if self.use_bias:
            y += tu.match_dims(x=self.b, dims=y.ndim - 1)

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                dy = dy.astype(self.w.dtype)
                self.set_dy(dy)

                w_y, w_x = self.w.shape[-2:]
                x_y, x_x = x.shape[-2:]
                dy_b, dy_k, dy_y, dy_x = dy.shape
                s_y, s_x = self.stride
                d_y, d_x = self.dil

                # fill elements skipped by strides with zeros
                dy_p_shape = (dy_b, dy_k, s_y * dy_y, s_x * dy_x)
                dy_p = tu.zeros(dy_p_shape, device=self.device).data
                dy_p[:, :, ::s_y, ::s_x] = dy
                dy_p_y = 1 + (x_y - w_y) if self.pad == "valid" else x_y
                dy_p_x = 1 + (x_x - w_x) if self.pad == "valid" else x_x
                dy_p = Tensor(dy_p[:, :, :dy_p_y, :dy_p_x], device=self.device)

                # input grads (b, c_in, y, x)
                dy_p_ext, w_ext = tu.expand_dims(dy_p, 2), tu.expand_dims(self.w, 0)
                pad = "full" if self.pad == "valid" else "same"
                # convolve (b, c_out, 1, y, x) * (1, c_out, c_in, y, x)
                dy_conv_w = convolve2d(dy_p_ext, w_ext, dil=self.dil, pad=pad)
                dy_conv_w = dy_conv_w.astype(self.dtype)  # conv returns float64
                dx = dy_conv_w.sum(axis=1).data  # sum over c_out

                # weight grads (c_out, c_in, y, x)
                dy_p_ext = dy_p_ext.flip((-2, -1))
                pad = (
                    (w_y // 2 * d_y, w_x // 2 * d_x) if self.pad == "same" else "valid"
                )
                # convolve (b, 1, c_in, y, x) * (b, c_out, 1, y, x)
                x_conv_dy = convolve2d(x_ext, dy_p_ext, pad=pad)[
                    :, :, :, -w_y * d_y :, -w_x * d_x :
                ]
                x_conv_dy = x_conv_dy.astype(self.dtype)  # conv returns float64
                # sum over b
                self.w.grad = x_conv_dy[:, :, :, ::d_y, ::d_x].sum(axis=0).data

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
        p_y, p_x = self.kernel_size
        x_b, x_c, x_y, x_x = x.shape

        # crop input to be a multiple of the pooling window size
        y_y = x_y // p_y * p_y
        y_x = x_x // p_x * p_x
        x_crop = x[:, :, :y_y, :y_x]

        # initialize with zeros
        y_shape = (x_b, x_c, y_y // p_y, y_x // p_x)
        y = tu.zeros(y_shape, dtype=x.dtype, device=x.device)

        # iterate over height and width and pick highest value
        for i in range(y.shape[-2]):
            for j in range(y.shape[-1]):
                c = x.data[:, :, i * p_y : (i + 1) * p_y, j * p_x : (j + 1) * p_x]
                y[:, :, i, j] = c.max(axis=(-2, -1))

        if self.training:
            # create map of max value occurences for backprop
            y_stretched = stretch2d(y, self.kernel_size, x_crop.shape)
            p_map = (x_crop == y_stretched) * 1.0

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)

                # stretch dy tensor to original shape by duplicating values
                dy_s = stretch2d(
                    Tensor(dy, dtype=dy.dtype, device=self.device),
                    self.kernel_size,
                    p_map.shape,
                )

                # use p_map as mask for grads
                return (dy_s * p_map).resize(x.shape).data

            self.backward = backward

        self.set_y(y)
        return y
