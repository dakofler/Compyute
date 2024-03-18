"""Convolution layers module"""

from compyute.functional import prod, zeros
from compyute.nn.funcional import convolve1d, convolve2d, stretch2d
from compyute.nn.module import Module
from compyute.nn.parameter import Parameter
from compyute.random import uniform
from compyute.tensor import Tensor


__all__ = ["Convolution1d", "Convolution2d", "MaxPooling2d", "AvgPooling2d"]


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
        use_bias: bool = True,
        dtype: str = "float32",
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
            Number of output channels.
        kernel_size : int
            Shape of each kernel.
        pad: str, optional
            Padding applied before convolution.
            Options are "causal", "valid" or "same", by default "causal".
        stride : int, optional
            Stride used for the convolution operation, by default 1.
        dil : int, optional
            Dilation used for each axis of the filter, by default 1.
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

        # init weights
        # (Co, Ci, K)
        k = (in_channels * kernel_size) ** -0.5
        w = uniform((out_channels, in_channels, kernel_size), -k, k)
        self.w = Parameter(w, dtype=dtype, label="w")

        # init biases
        # (Co,)
        if use_bias:
            b = zeros((out_channels,))
            self.b = Parameter(b, dtype=dtype, label="b")

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

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [3])
        x = x.astype(self.dtype)

        # flip weights for cross correlation
        w_flip = self.w.flip(-1)

        x = x.insert_dim(axis=1)  # (B, 1, Ci, Ti)
        w_flip = w_flip.reshape((1, *w_flip.shape))  # (1, Co, Ci, K)

        # convolve
        # (B, 1, Ci, Ti) * (1, Co, Ci, K) -> (B, Co, Ci, To)
        x_conv_w = convolve1d(x, w_flip, self.stride, self.dil, self.pad)

        # sum over input channels
        # (B, Co, Ci, To) -> (B, Co, To)
        y = x_conv_w.sum(axis=2)

        if self.use_bias:
            # (B, Co, To) + (Co, 1)
            y += self.b.reshape((*self.b.shape, 1))

        if self.training:
            def backward(dy: Tensor) -> Tensor:
                dy = dy.astype(self.dtype)
                self.set_dy(dy)

                K = self.w.shape[-1]
                Ti = x.shape[-1]
                B, Co, To = dy.shape

                # undo strides by filling with zeros
                dy_p = zeros((B, Co, self.stride * To), device=self.device)
                dy_p[:, :, :: self.stride] = dy
                dy_p_ti = 1 + (Ti - K) if self.pad == "valid" else Ti
                dy_p = dy_p[:, :, :dy_p_ti]

                # ----------------
                # input grads
                # ----------------
                dy_p_ext = dy_p.insert_dim(axis=2)  # (B, Co, 1, To)
                w_ext = self.w.reshape((1, *self.w.shape))  # (1, Co, Ci, K)
                pad = "full" if self.pad == "valid" else self.pad

                # convolve
                # (B, Co, 1, To) * (1, Co, Ci, K)
                dy_conv_w = convolve1d(dy_p_ext, w_ext, dil=self.dil, pad=pad)

                # sum over output channels
                # (B, Ci, Ti)
                dx = dy_conv_w.sum(axis=1)

                # ----------------
                # weight grads
                # ----------------
                dy_p_ext = dy_p_ext.flip(-1)

                match self.pad:
                    case "same":
                        pad = K // 2 * self.dil
                    case "causal":
                        pad = (K // 2 * self.dil * 2, 0)
                    case _:
                        pad = self.pad

                # convolve
                # (B, 1, Ci, Ti) * (B, Co, 1, To) -> (B, Co, Ci, K)
                x_conv_dy = convolve1d(x, dy_p_ext, pad=pad)[
                    :, :, :, -K * self.dil:]

                # sum over batches
                # (B, Co, Ci, K) -> (Co, Ci, K)
                self.w.grad = x_conv_dy[:, :, :, :: self.dil].sum(axis=0)

                # ----------------
                # bias grads
                # ----------------
                if self.use_bias:
                    # sum over batches and time
                    # (B, Co, To) -> (Co,)
                    self.b.grad = dy.sum(axis=(0, 2))

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
        use_bias: bool = True,
        dtype: str = "float32",
    ) -> None:
        """Convolutional layer used for spacial information and feature extraction.
        Input: (B, Ci, Yi, Xi)
            B ... batch, Ci ... input channels, Yi ... input height, Xi ... input width
        Output: (B, Co, Yo, Xo)
            B ... batch, Co ... output channels, Yo ... output height, Xo ... output width

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

        # init weights
        # (Co, Ci, Ky, Kx)
        k = (in_channels * prod(kernel_size)) ** -0.5
        w = uniform((out_channels, in_channels, *kernel_size), -k, k)
        self.w = Parameter(w, dtype=dtype, label="w")

        # init biases
        # (Co,)
        if self.use_bias:
            self.b = Parameter(zeros((out_channels,)), dtype=dtype, label="b")

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

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [4])
        x = x.astype(self.dtype)

        # rotate weights for cross correlation
        w_flip = self.w.flip((-2, -1))

        x = x.insert_dim(axis=1)  # (B, 1, Ci, Yi, Xi)
        w_flip = w_flip.reshape((1, *w_flip.shape))  # (1, Co, Ci, Ky, Kx)

        # convolve
        # (B, 1, Ci, Yi, Xi) * (1, Co, Ci, Ky, Kx) -> (B, Co, Ci, Yo, Xo)
        x_conv_w = convolve2d(x, w_flip, self.stride, self.dil, self.pad)

        # sum over input channels
        # (B, Co, Ci, Yo, Xo) -> (B, Co, Yo, Xo)
        y = x_conv_w.sum(axis=2)

        if self.use_bias:
            # (B, Co, Yo, Xo) + (Co, 1, 1)
            y += self.b.add_dims(target_dims=3)

        if self.training:
            def backward(dy: Tensor) -> Tensor:
                dy = dy.astype(self.dtype)
                self.set_dy(dy)

                Ky, Kx = self.w.shape[-2:]
                Yi, Xi = x.shape[-2:]
                B, Co, Yo, Xo = dy.shape
                Sy, Sx = self.stride
                Dy, Dx = self.dil

                # fill elements skipped by strides with zeros
                dy_p = zeros((B, Co, Sy * Yo, Sx * Xo), device=self.device)
                dy_p[:, :, ::Sy, ::Sx] = dy
                dy_p_yi = 1 + (Yi - Ky) if self.pad == "valid" else Yi
                dy_p_xi = 1 + (Xi - Kx) if self.pad == "valid" else Xi
                dy_p = dy_p[:, :, :dy_p_yi, :dy_p_xi]

                # ----------------
                # input grads
                # ----------------
                dy_p_ext = dy_p.insert_dim(axis=2)  # (B, Co, 1, Yo, Xo)
                # (1, Co, Ci, Ky, Kx)
                w_ext = self.w.reshape((1, *self.w.shape))
                pad = "full" if self.pad == "valid" else "same"

                # convolve
                # (B, Co, 1, Yo, Xo) * (1, Co, Ci, Ky, Kx) -> (B, Co, Ci, Yi, Xi)
                dy_conv_w = convolve2d(dy_p_ext, w_ext, dil=self.dil, pad=pad)

                # sum over c_out
                # (B, Co, Ci, Yi, Xi) -> (B, Ci, Yi, Xi)
                dx = dy_conv_w.sum(axis=1)

                # ----------------
                # weight grads
                # ----------------
                dy_p_ext = dy_p_ext.flip((-2, -1))

                pad = (Ky // 2 * Dy, Kx // 2 *
                       Dx) if self.pad == "same" else "valid"

                # convolve
                # (B, 1, Ci, Yi, Xi) * (B, Co, 1, Yo, Xo) -> (B, Co, Ci, Ky, Kx)
                x_conv_dy = convolve2d(x, dy_p_ext, pad=pad)[
                    :, :, :, -Ky * Dy:, -Kx * Dx:
                ]

                # sum over batches
                # (B, Co, Ci, Ky, Kx) -> (Co, Ci, Ky, Kx)
                self.w.grad = x_conv_dy[:, :, :, ::Dy, ::Dx].sum(axis=0)

                # ----------------
                # bias grads
                # ----------------
                if self.use_bias:
                    # sum over batches, height and width
                    # (B, Co, Yo, Xo) -> (Co,)
                    self.b.grad = dy.sum(axis=(0, 2, 3))

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

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [4])
        ky, kx = self.kernel_size
        b, c, yi, xi = x.shape

        # crop input to be a multiple of the pooling window size
        yo = yi // ky * ky
        xo = xi // kx * kx
        x_crop = x[:, :, :yo, :xo]

        # initialize with zeros
        y = zeros((b, c, yo // ky, xo // kx), dtype=x.dtype, device=x.device)

        # iterate over height and width and pick highest value
        for i in range(y.shape[-2]):
            for j in range(y.shape[-1]):
                chunk = x.data[:, :, i *
                               ky: (i + 1) * ky, j * kx: (j + 1) * kx]
                y[:, :, i, j] = chunk.max(axis=(-2, -1))

        if self.training:
            # create map of max value occurences for backprop
            y_stretched = stretch2d(y, self.kernel_size, x_crop.shape)
            p_map = (x_crop == y_stretched).int()

            def backward(dy: Tensor) -> Tensor:
                self.set_dy(dy)

                # stretch dy tensor to original shape by duplicating values
                dy_str = stretch2d(dy, self.kernel_size, p_map.shape)

                # use p_map as mask for grads
                dx = dy_str * p_map
                return dx if dx.shape == x.shape else dx.pad_to_shape(x.shape)
            self.backward = backward

        self.set_y(y)
        return y


class AvgPooling2d(Module):
    """AvgPooling layer used to reduce information to avoid overfitting."""

    def __init__(
        self, kernel_size: tuple[int, int] = (2, 2), dtype: str = "float32"
    ) -> None:
        """AvgPooling layer used to reduce information to avoid overfitting.

        Parameters
        ----------
        kernel_size : tuple[int, int], optional
             Shape of the pooling window used for the pooling operation, by default (2, 2).
        dtype: str, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.dtype = dtype

    def __repr__(self) -> str:
        name = self.__class__.__name__
        kernel_size = self.kernel_size
        return f"{name}({kernel_size=})"

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [4])
        Ky, Kx = self.kernel_size
        B, C, Yi, Xi = x.shape

        # crop input to be a multiple of the pooling window size
        yo = Yi // Ky * Ky
        xo = Xi // Kx * Kx
        x_crop = x[:, :, :yo, :xo]

        # initialize with zeros
        y = zeros((B, C, yo // Ky, xo // Kx), dtype=x.dtype, device=x.device)

        # iterate over height and width and compute mean value
        for i in range(y.shape[-2]):
            for j in range(y.shape[-1]):
                chunk = x.data[:, :, i *
                               Ky: (i + 1) * Ky, j * Kx: (j + 1) * Kx]
                y[:, :, i, j] = chunk.mean(axis=(-2, -1))

        if self.training:
            def backward(dy: Tensor) -> Tensor:
                self.set_dy(dy)

                # stretch dy tensor to original shape by duplicating values
                dy_str = stretch2d(dy, self.kernel_size, x_crop.shape)

                # scale gradients down
                dx = dy_str / prod(self.kernel_size)
                return dx if dx.shape == x.shape else dx.pad_to_shape(x.shape)
            self.backward = backward

        self.set_y(y)
        return y
