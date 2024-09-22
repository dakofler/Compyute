"""Neural network convolution functions."""

from typing import Optional

from ...tensor_ops.creation_ops import zeros
from ...tensor_ops.multiary_ops import einsum
from ...tensor_ops.reshape_ops import (
    flip,
    insert_dim,
    pad,
    pad_to_shape,
    pooling1d,
    pooling2d,
    repeat,
)
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


class Convolution1DFn(Function):
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
        f = Dilation1DFn.forward(cache, f, dilation)
        x = Pad1DFn.forward(cache, x, padding)
        y = _Convolution1DFn.forward(cache, x, f, stride)
        if b:
            y += insert_dim(b, -1)

        cache.push(b is not None)
        return y

    @staticmethod
    def backward(
        cache: FunctionCache, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        (b,) = cache.pop()

        dx, df = _Convolution1DFn.backward(cache, dy)
        dx = Pad1DFn.backward(cache, dx)
        df = Dilation1DFn.backward(cache, df)
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
    :class:`compyute.nn.Convolution1D`
    """
    return Convolution1DFn.forward(PseudoCache(), x, f, b, padding, stride, dilation)


class Dilation1DFn(Function):
    """Dilates a tensor in its last axis."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, dilation: int) -> Tensor:
        no_dilation = dilation == 1
        cache.push(no_dilation, dilation)
        if no_dilation:
            return x

        dil_shape = (dilation * x.shape[-1] - 1,)
        y = zeros(x.shape[:-1] + dil_shape, device=x.device, dtype=x.dtype)
        y[..., ::dilation] = x

        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        no_dilation, dilation = cache.pop()
        if no_dilation:
            return dy
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
    return Dilation1DFn.forward(PseudoCache(), x, dilation)


class Pad1DFn(Function):
    """Pads a tensor in its last axis."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, padding: int) -> Tensor:
        no_padding = padding == 0
        cache.push(no_padding, padding)
        if no_padding:
            return x

        widths = tuple([(0, 0)] * (x.n_axes - 1) + [(padding, padding)])
        y = pad(x, widths)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        no_padding, padding = cache.pop()
        if no_padding:
            return dy
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
    return Pad1DFn.forward(PseudoCache(), x, padding)


class _Convolution1DFn(Function):
    """Computes the 1D convolution of two tensors."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, f: Tensor, stride: int) -> Tensor:
        x_pooled = pooling1d(x, f.shape[-1], stride)  # reshape x to (B, Ci, So, F)
        y = einsum("bitf,oif->bot", x_pooled, f)  # multiply and add
        cache.push(x, f, stride)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> tuple[Tensor, Tensor]:
        x, f, stride = cache.pop()

        # fill elements skipped by strides with zeros
        dy = dilate1d(dy, stride)

        # pad to match unstrided dy
        dy_t = x.shape[-1] - f.shape[-1] + 1
        dy = pad_to_shape(dy, (*dy.shape[:-1], dy_t))

        # full pad
        dy = pad1d(dy, f.shape[-1] - 1)

        # input grads
        dy_pooled = pooling1d(dy, f.shape[-1])  # reshape to (B, Co, Si, F)
        f = flip(f, axis=-1)
        dx = einsum("bosf,oif->bis", dy_pooled, f)

        # filter grads
        dy_pooled = pooling1d(dy, x.shape[-1])  # reshape to (B, Co, F, Si)
        df = einsum("bofs,bis->oif", dy_pooled, x)
        df = flip(df, axis=-1)

        return dx, df


class Convolution2DFn(Function):
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
        f = Dilation2DFn.forward(cache, f, dilation)
        x = Pad2DFn.forward(cache, x, padding)
        y = _Convolution2DFn.forward(cache, x, f, stride)
        if b:
            y += b.to_shape((*b.shape, 1, 1))

        cache.push(b is not None)
        return y

    @staticmethod
    def backward(
        cache: FunctionCache, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        (b,) = cache.pop()

        dx, df = _Convolution2DFn.backward(cache, dy)
        dx = Pad2DFn.backward(cache, dx)
        df = Dilation2DFn.backward(cache, df)
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
    :class:`compyute.nn.Convolution2D`
    """
    return Convolution2DFn.forward(PseudoCache(), x, f, b, padding, stride, dilation)


class Dilation2DFn(Function):
    """Dilates a tensor in its last two axes."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, dilation: int) -> Tensor:
        no_dilation = dilation == 1
        cache.push(no_dilation, dilation)
        if no_dilation:
            return x

        dil_shape = (dilation * x.shape[-2] - 1, dilation * x.shape[-1] - 1)
        y = zeros(x.shape[:-2] + dil_shape, device=x.device, dtype=x.dtype)
        y[..., ::dilation, ::dilation] = x

        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        no_dilation, dilation = cache.pop()
        if no_dilation:
            return dy
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
    return Dilation2DFn.forward(PseudoCache(), x, dilation)


class Pad2DFn(Function):
    """Pads a tensor in its last two axes."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, padding: int) -> Tensor:
        no_padding = padding == 0
        cache.push(no_padding, padding)
        if no_padding:
            return x
        widths = tuple([(0, 0)] * (x.n_axes - 2) + [(padding, padding)] * 2)
        y = pad(x, widths)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        no_padding, padding = cache.pop()
        if no_padding:
            return dy
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
    return Pad2DFn.forward(PseudoCache(), x, padding)


class _Convolution2DFn(Function):
    """Computes the 2D convolution of two tensors."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, f: Tensor, stride: int) -> Tensor:
        x1_pooled = pooling2d(
            x, f.shape[-1], stride
        )  # reshape to (B, Ci, Y, X, Fy, Fx)
        y = einsum("biyxjk,oijk->boyx", x1_pooled, f)  # multiply and add
        cache.push(x, f, stride)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> tuple[Tensor, Tensor]:
        x, f, stride = cache.pop()

        # fill elements skipped by strides with zeros
        dy = dilate2d(dy, stride)

        # pad to match unstrided dy
        dy_t = x.shape[-1] - f.shape[-1] + 1
        dy = pad_to_shape(dy, (*dy.shape[:-2], dy_t, dy_t))

        # full pad
        dy = pad2d(dy, f.shape[-1] - 1)

        # input grads
        dy_pooled = pooling2d(dy, f.shape[-1])  # reshape to (B, Co, Y, X, Fy, Fx)
        f = flip(f, axis=(-2, -1))
        dx = einsum("boyxjk,oijk->biyx", dy_pooled, f)

        # filter grads
        dy_pooled = pooling2d(dy, x.shape[-1])  # reshape to (B, Co, Fy, Fx, Y, X)
        df = einsum("bojkyx,biyx->oijk", dy_pooled, x)
        df = flip(df, axis=(-2, -1))

        return dx, df


class Upsample2DFn(Function):
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
    return Upsample2DFn.forward(x, scaling, shape)


class MaxPooling2DFn(Function):
    """Performs max pooling over the last two axes."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, kernel_size: int) -> Tensor:
        y = pooling2d(x, kernel_size, kernel_size).max(axis=(-2, -1))
        cache.push(x, kernel_size, y)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, kernel_size, y = cache.pop()
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
    return MaxPooling2DFn.forward(PseudoCache(), x, kernel_size)


class AvgPooling2DFn(Function):
    """Performs average pooling over the last two axes."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, kernel_size: int) -> Tensor:
        y = pooling2d(x, kernel_size, kernel_size).mean(axis=(-2, -1))
        cache.push(x.shape, kernel_size)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x_shape, kernel_size = cache.pop()
        return upsample2d(dy / (kernel_size * kernel_size), kernel_size, x_shape)


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
    return AvgPooling2DFn.forward(PseudoCache(), x, kernel_size)
