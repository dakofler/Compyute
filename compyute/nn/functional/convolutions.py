"""Neural network convolution functions."""

from typing import Optional

from ...tensor_ops.creating import zeros
from ...tensor_ops.reshaping import (
    broadcast_to,
    flip,
    insert_dim,
    pad,
    pad_to_shape,
    repeat,
)
from ...tensor_ops.transforming import convolve1d_fft, convolve2d_fft
from ...tensors import ShapeLike, Tensor
from .functions import Function, FunctionCache, PseudoCache

__all__ = [
    "convolve1d",
    "dilate1d",
    "pad1d",
    "convolve2d",
    "dilate2d",
    "pad2d",
    "upsample2d",
    "maxpooling2d",
    "avgpooling2d",
]


class FConvolution1D(Function):
    """Computes the convolution of two tensors over their last axis."""

    @staticmethod
    def forward(
        cache: FunctionCache,
        x: Tensor,
        f: Tensor,
        b: Optional[Tensor],
        padding: int,
        stride: int,
        dilation: int,
    ) -> Tensor:
        # dilate filter and add a fake batch dimension
        f = FDilation1D.forward(cache, f, dilation)
        f = insert_dim(f, 0)  # (1, Co, Ci, F)

        # pad input and add a fake output dimension
        x = FPad1D.forward(cache, x, padding)
        x = insert_dim(x, 1)  # (B, 1, Ci, T)

        # perform convolution and sum over input dimension (B, Co, Ci, T)
        conv = _FConvolution1D.forward(cache, x, f, stride)
        y = conv.sum(axis=2)  # (B, Co, T)

        if b:
            y += insert_dim(b, -1)

        cache.b, cache.conv_shape = b is not None, conv.shape
        return y

    @staticmethod
    def backward(
        cache: FunctionCache, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        b, conv_shape = cache.b, cache.conv_shape

        # insert fake input channel dimension
        dy_ext = insert_dim(dy, 2)  # (B, Co, 1, X)
        dy_ext = broadcast_to(dy_ext, conv_shape)  # (B, Co, Ci, X)
        dx, df = _FConvolution1D.backward(cache, dy_ext)

        dx = FPad1D.backward(cache, dx.sum(axis=1))
        df = FDilation1D.backward(cache, df.sum(axis=0))
        db = None if not b else dy.sum(axis=(0, 2))

        return dx, df, db


def convolve1d(
    x: Tensor,
    f: Tensor,
    b: Optional[Tensor] = None,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
) -> Tensor:
    """Computes the convolution of two tensors over their last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    b : Tensor, optional
        Bias tensor. Defaults to ``None``. If ``None``, no bias is added.
    padding : int, optional
        Padding applied to the input tensor. Defaults to ``0``.
    stride : int, optional
        Stride used in the convolution operation. Defaults to ``1``.
    dilation : int, optional
        Dilation factor to use for each axis of the filter. Defaults to ``1``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.Convolution1d`
    """
    return FConvolution1D.forward(PseudoCache(), x, f, b, padding, stride, dilation)


class FDilation1D(Function):
    """Dilates a tensor in its last axis."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, dilation: int) -> Tensor:
        no_dilation = dilation == 1
        cache.no_dilation = no_dilation
        if no_dilation:
            return x

        dil_shape = (dilation * x.shape[-1] - 1,)
        y = zeros(x.shape[:-1] + dil_shape, device=x.device, dtype=x.dtype)
        y[..., ::dilation] = x

        cache.dilation = dilation
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        no_dilation = cache.no_dilation
        if no_dilation:
            return dy
        dilation = cache.dilation
        return dy[..., ::dilation]


def dilate1d(x: Tensor, dilation: int) -> Tensor:
    """Dilates a tensor in its last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dilation : int
        Dilation factor to use.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return FDilation1D.forward(PseudoCache(), x, dilation)


class FPad1D(Function):
    """Pads a tensor in its last axis."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, padding: int) -> Tensor:
        no_padding = padding == 0
        cache.no_padding = no_padding
        if no_padding:
            return x

        widths = tuple([(0, 0)] * (x.n_axes - 1) + [(padding, padding)])
        y = pad(x, widths)
        cache.padding = padding
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        no_padding = cache.no_padding
        if no_padding:
            return dy
        padding = cache.padding
        return dy[..., padding:-padding]


def pad1d(x: Tensor, padding: int) -> Tensor:
    """Pads a tensor in its last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    padding : int
        Padding width applied to the beginning and end of the last axis.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return FPad1D.forward(PseudoCache(), x, padding)


class _FConvolution1D(Function):
    """Computes the 1D convolution of two tensors."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, f: Tensor, stride: int) -> Tensor:
        conv = convolve1d_fft(x, flip(f, -1))
        y = conv[..., ::stride]

        cache.x, cache.f, cache.stride, cache.conv_shape = x, f, stride, conv.shape
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> tuple[Tensor, Tensor]:
        x, f, stride, conv_shape = cache.x, cache.f, cache.stride, cache.conv_shape

        # fill elements skipped by strides with zeros
        dy = pad_to_shape(dilate1d(dy, stride), conv_shape)
        dy = pad1d(dy, f.shape[-1] - 1)  # full pad dy
        dx = convolve1d_fft(dy, f)
        df = convolve1d_fft(flip(dy, axis=-1), x)

        return dx, df


class FConvolution2D(Function):
    """Computes the convolution of two tensors over their last axis."""

    @staticmethod
    def forward(
        cache: FunctionCache,
        x: Tensor,
        f: Tensor,
        b: Optional[Tensor],
        padding: int,
        stride: int,
        dilation: int,
    ) -> Tensor:
        # dilate filter and add a fake batch dimension
        f = FDilation2D.forward(cache, f, dilation)
        f = insert_dim(f, 0)  # (1, Co, Ci, Fy, Fx)

        # pad input and add a fake output dimension
        x = FPad2D.forward(cache, x, padding)
        x = insert_dim(x, 1)  # (B, 1, Ci, Y, X)

        # perform convolution and sum over input dimension (B, Co, Ci, T)
        conv = _FConvolution2D.forward(cache, x, f, stride)
        y = conv.sum(axis=2)  # (B, Co, Y, X)

        if b:
            y += b.to_shape((*b.shape, 1, 1))

        cache.b, cache.conv_shape = b is not None, conv.shape
        return y

    @staticmethod
    def backward(
        cache: FunctionCache, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        b, conv_shape = cache.b, cache.conv_shape

        # insert fake input channel dimension
        dy_ext = insert_dim(dy, 2)  # (B, Co, 1, Y, X)
        dy_ext = broadcast_to(dy_ext, conv_shape)  # (B, Co, Ci, Y, X)
        dx, df = _FConvolution2D.backward(cache, dy_ext)

        dx = FPad2D.backward(cache, dx.sum(axis=1))
        df = FDilation2D.backward(cache, df.sum(axis=0))
        db = None if not b else dy.sum(axis=(0, 2, 3))

        return dx, df, db


def convolve2d(
    x: Tensor,
    f: Tensor,
    b: Optional[Tensor] = None,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
) -> Tensor:
    """Computes the convolution of two tensors over their last two axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    b : Tensor, optional
        Bias tensor. Defaults to ``None. If ``None``, no bias is added.
    padding : int, optional
        Padding applied to the input tensor. Defaults to ``0``.
    stride : int, optional
        Stride used in the convolution operation. Defaults to ``1``.
    dilation : int, optional
        Dilation factor to use for each axis of the filter. Defaults to ``1``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.Convolution2d`
    """
    return FConvolution2D.forward(PseudoCache(), x, f, b, padding, stride, dilation)


class FDilation2D(Function):
    """Dilates a tensor in its last two axes."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, dilation: int) -> Tensor:
        no_dialution = dilation == 1
        cache.no_dilation = no_dialution
        if no_dialution:
            return x

        dil_shape = (dilation * x.shape[-2] - 1, dilation * x.shape[-1] - 1)
        y = zeros(x.shape[:-2] + dil_shape, device=x.device, dtype=x.dtype)
        y[..., ::dilation, ::dilation] = x

        cache.dilation = dilation
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        no_dilation = cache.no_dilation
        if no_dilation:
            return dy
        dilation = cache.dilation
        return dy[..., ::dilation, ::dilation]


def dilate2d(x: Tensor, dilation: int) -> Tensor:
    """Dilates a tensor in its last two axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dilation : int
        Dilation factor to use.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return FDilation2D.forward(PseudoCache(), x, dilation)


class FPad2D(Function):
    """Pads a tensor in its last two axes."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, padding: int) -> Tensor:
        no_padding = padding == 0
        cache.no_padding = no_padding
        if no_padding:
            return x
        widths = tuple([(0, 0)] * (x.n_axes - 2) + [(padding, padding)] * 2)
        y = pad(x, widths)
        cache.padding = padding
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        no_padding = cache.no_padding
        if no_padding:
            return dy
        padding = cache.padding
        return dy[..., padding:-padding, padding:-padding]


def pad2d(x: Tensor, padding: int) -> Tensor:
    """Pads a tensor in its last two axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    padding : int
        Padding width applied to the beginning and end of the last two axes.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return FPad2D.forward(PseudoCache(), x, padding)


class _FConvolution2D(Function):
    """Computes the 2D convolution of two tensors."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, f: Tensor, strides: int) -> Tensor:
        conv = convolve2d_fft(x, flip(f, (-2, -1)))
        y = conv[..., ::strides, ::strides]

        cache.x, cache.f, cache.strides, cache.conv_shape = x, f, strides, conv.shape
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> tuple[Tensor, Tensor]:
        x, f, strides, conv_shape = cache.x, cache.f, cache.strides, cache.conv_shape

        # fill elements skipped by strides with zeros
        dy = pad_to_shape(dilate2d(dy, strides), conv_shape)
        dy = pad2d(dy, f.shape[-1] - 1)  # full pad dy
        dx = convolve2d_fft(dy, f)
        df = convolve2d_fft(flip(dy, (-2, -1)), x)

        return dx, df


class FUpsample2D(Function):
    """Upsamples a tensor by repeating it's elements over the last two axes."""

    @staticmethod
    def forward(x: Tensor, scaling: int, shape: ShapeLike) -> Tensor:
        x = repeat(repeat(x, scaling, -1), scaling, -2)
        y = x if x.shape == shape else pad_to_shape(x, shape)
        return y


def upsample2d(x: Tensor, scaling: int, shape: ShapeLike) -> Tensor:
    """Upsamples a tensor by repeating it's elements over the last two axes.

    Parameters
    ----------
    x : Tensor
        Tensor to be stretched out.
    scaling : int
        Number of repeating values along each axis.
    shape : ShapeLike
        Shape of the target tensor. If the shape does not match after upsampling,
        remaining values are filled with zeroes.

    Returns
    -------
    Tensor
        Upsampled tensor.
    """
    return FUpsample2D.forward(x, scaling, shape)


class FMaxPooling2D(Function):
    """Performs max pooling over the last two axes."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, kernel_size: int) -> Tensor:
        x_height, x_width = x.shape[-2:]

        trunc_y = x_height // kernel_size * kernel_size
        trunc_x = x_width // kernel_size * kernel_size
        x_trunc = x[..., :trunc_y, :trunc_x]
        pool_shape = x.shape[:-2] + (
            x_height // kernel_size,
            kernel_size,
            x_width // kernel_size,
            kernel_size,
        )
        y = x_trunc.to_shape(pool_shape).max(axis=(-3, -1))

        cache.x, cache.kernel_size, cache.y = x, kernel_size, y
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, kernel_size, y = cache.x, cache.kernel_size, cache.y
        y_ups = upsample2d(y, kernel_size, x.shape)
        return upsample2d(dy, kernel_size, x.shape) * (x == y_ups)


def maxpooling2d(x: Tensor, kernel_size: int = 2) -> Tensor:
    """Performs max pooling over the last two axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    kernel_size : int, optional
        Size of the pooling window. Defaults to ``2``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.MaxPooling2D`
    """
    return FMaxPooling2D.forward(PseudoCache(), x, kernel_size)


class FAvgPooling2D(Function):
    """Performs average pooling over the last two axes."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, kernel_size: int) -> Tensor:
        x_height, x_width = x.shape[-2:]

        trunc_y = x_height // kernel_size * kernel_size
        trunc_x = x_width // kernel_size * kernel_size
        x_trunc = x[..., :trunc_y, :trunc_x]

        pool_shape = x.shape[:-2] + (
            x_height // kernel_size,
            kernel_size,
            x_width // kernel_size,
            kernel_size,
        )
        y = x_trunc.to_shape(pool_shape).mean(axis=(-3, -1))

        cache.x_shape, cache.kernel_size = x.shape, kernel_size
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x_shape, kernel_size = cache.x_shape, cache.kernel_size
        return upsample2d(dy / kernel_size**2, kernel_size, x_shape)


def avgpooling2d(x: Tensor, kernel_size: int = 2) -> Tensor:
    """Performs average pooling over the last two axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    kernel_size : int, optional
        Size of the pooling window. Defaults to ``2``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.AvgPooling2D`
    """
    return FAvgPooling2D.forward(PseudoCache(), x, kernel_size)
