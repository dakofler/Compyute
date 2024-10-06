"""Neural network convolution functions."""

from typing import Optional

from ...tensor_ops.creation_ops import zeros
from ...tensor_ops.multiary_ops import einsum
from ...tensor_ops.shape_ops import flip, pad, pad_to_shape, pooling1d, pooling2d
from ...tensors import ShapeError, Tensor
from .functions import Function, FunctionCache, PseudoCache

__all__ = [
    "conv1d",
    "dilate1d",
    "pad1d",
    "conv2d",
    "dilate2d",
    "pad2d",
    "conv_transpose1d",
    "conv_transpose2d",
]


class Conv1DFn(Function):
    """Computes the convolution of two tensors over their last dimension."""

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
        if x.ndim != 3:
            raise ShapeError(f"Expected input to be 3D, got {x.ndim}D.")

        f = Dilation1DFn.forward(cache, f, dilation)
        x = Pad1DFn.forward(cache, x, padding)
        y = RawConv1DFn.forward(cache, x, f, stride)
        if b:
            y += b.view((*b.shape, 1))

        cache.push(b is not None)
        return y

    @staticmethod
    def backward(
        cache: FunctionCache, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        (b,) = cache.pop()

        dx, df = RawConv1DFn.backward(cache, dy)
        dx = Pad1DFn.backward(cache, dx)
        df = Dilation1DFn.backward(cache, df)
        db = None if not b else dy.sum((0, 2))

        return dx, df, db


def conv1d(
    x: Tensor,
    f: Tensor,
    b: Optional[Tensor] = None,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
) -> Tensor:
    """Computes the convolution of two tensors over their last dimension.

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
        Dilation factor used for the filter. Defaults to ``1``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.Convolution1D`
    """
    return Conv1DFn.forward(PseudoCache(), x, f, b, padding, stride, dilation)


class Dilation1DFn(Function):
    """Dilates a tensor in its last dimension."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, dilation: int) -> Tensor:
        no_dilation = dilation == 1
        cache.push(no_dilation, dilation)
        if no_dilation:
            return x

        y_shape = (*x.shape[:-1], dilation * (x.shape[-1] - 1) + 1)
        y = zeros(y_shape, device=x.device, dtype=x.dtype)
        y[..., ::dilation] = x

        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        no_dilation, dilation = cache.pop()
        if no_dilation:
            return dy
        return dy[..., ::dilation]


def dilate1d(x: Tensor, dilation: int) -> Tensor:
    """Dilates a tensor in its last dimension.

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
    """Pads a tensor in its last dimension."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, padding: int) -> Tensor:
        no_padding = padding == 0
        cache.push(no_padding, padding)
        if no_padding:
            return x

        widths = tuple([(0, 0)] * (x.ndim - 1) + [(padding, padding)])
        y = pad(x, widths)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        no_padding, padding = cache.pop()
        if no_padding:
            return dy
        return dy[..., padding:-padding].to_contiguous()


def pad1d(x: Tensor, padding: int) -> Tensor:
    """Pads a tensor in its last dimension.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    padding : int
        Padding width applied to the beginning and end of the last dimension.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return Pad1DFn.forward(PseudoCache(), x, padding)


class RawConv1DFn(Function):
    """Computes the 1D convolution of two tensors."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, f: Tensor, stride: int) -> Tensor:
        x_pooled = pooling1d(x, f.shape[-1], stride)  # view as (B, Ci, So, F)
        y = einsum("bitf,oif->bot", x_pooled, f).to_contiguous()  # multiply and add
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
        dy_pooled = pooling1d(dy, f.shape[-1])  # view as (B, Co, Si, F)
        f = flip(f, dim=-1)
        dx = einsum("bosf,oif->bis", dy_pooled, f).to_contiguous()

        # filter grads
        dy_pooled = pooling1d(dy, x.shape[-1])  # view as (B, Co, F, Si)
        df = einsum("bofs,bis->oif", dy_pooled, x)
        df = flip(df, dim=-1).to_contiguous()

        return dx, df


class Conv2DFn(Function):
    """Computes the convolution of two tensors over their last dimension."""

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
        if x.ndim != 4:
            raise ShapeError(f"Expected input to be 4D, got {x.ndim}D.")

        f = Dilation2DFn.forward(cache, f, dilation)
        x = Pad2DFn.forward(cache, x, padding)
        y = RawConv2DFn.forward(cache, x, f, stride)
        if b:
            y += b.view((*b.shape, 1, 1))

        cache.push(b is not None)
        return y

    @staticmethod
    def backward(
        cache: FunctionCache, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        (b,) = cache.pop()

        dx, df = RawConv2DFn.backward(cache, dy)
        dx = Pad2DFn.backward(cache, dx)
        df = Dilation2DFn.backward(cache, df)
        db = None if not b else dy.sum((0, 2, 3))

        return dx, df, db


def conv2d(
    x: Tensor,
    f: Tensor,
    b: Optional[Tensor] = None,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
) -> Tensor:
    """Computes the convolution of two tensors over their last two dimensions.

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
        Dilation factor used for the filter. Defaults to ``1``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.Convolution2D`
    """
    return Conv2DFn.forward(PseudoCache(), x, f, b, padding, stride, dilation)


class Dilation2DFn(Function):
    """Dilates a tensor in its last two dimensions."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, dilation: int) -> Tensor:
        no_dilation = dilation == 1
        cache.push(no_dilation, dilation)
        if no_dilation:
            return x

        y_height = dilation * (x.shape[-2] - 1) + 1
        y_width = dilation * (x.shape[-1] - 1) + 1
        y = zeros((*x.shape[:-2], y_height, y_width), device=x.device, dtype=x.dtype)
        y[..., ::dilation, ::dilation] = x

        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        no_dilation, dilation = cache.pop()
        if no_dilation:
            return dy
        return dy[..., ::dilation, ::dilation]


def dilate2d(x: Tensor, dilation: int) -> Tensor:
    """Dilates a tensor in its last two dimensions.

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
    """Pads a tensor in its last two dimensions."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, padding: int) -> Tensor:
        no_padding = padding == 0
        cache.push(no_padding, padding)
        if no_padding:
            return x
        widths = tuple([(0, 0)] * (x.ndim - 2) + [(padding, padding)] * 2)
        y = pad(x, widths)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        no_padding, padding = cache.pop()
        if no_padding:
            return dy
        return dy[..., padding:-padding, padding:-padding].to_contiguous()


def pad2d(x: Tensor, padding: int) -> Tensor:
    """Pads a tensor in its last two dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    padding : int
        Padding width applied to the beginning and end of the last two dimensions.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return Pad2DFn.forward(PseudoCache(), x, padding)


class RawConv2DFn(Function):
    """Computes the 2D convolution of two tensors."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, f: Tensor, stride: int) -> Tensor:
        x_pooled = pooling2d(x, f.shape[-1], stride)  # view as (B, Ci, Y, X, Fy, Fx)
        y = einsum("biyxjk,oijk->boyx", x_pooled, f).to_contiguous()  # multiply and add
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
        dy_pooled = pooling2d(dy, f.shape[-1])  # view as (B, Co, Y, X, Fy, Fx)
        f = flip(f, dim=(-2, -1))
        dx = einsum("boyxjk,oijk->biyx", dy_pooled, f).to_contiguous()

        # filter grads
        dy_pooled = pooling2d(dy, x.shape[-1])  # view as (B, Co, Fy, Fx, Y, X)
        df = einsum("bojkyx,biyx->oijk", dy_pooled, x)
        df = flip(df, dim=(-2, -1)).to_contiguous()

        return dx, df


class InvPad1DFn(Function):
    """Removes cols from a tensor's last dimension."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, padding: int) -> Tensor:
        no_padding = padding == 0
        cache.push(no_padding, padding)
        if no_padding:
            return x
        return x[..., padding:-padding].to_contiguous()

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        no_padding, padding = cache.pop()
        if no_padding:
            return dy
        widths = tuple([(0, 0)] * (dy.ndim - 1) + [(padding, padding)])
        return pad(dy, widths)


class ConvTranspose1DFn(Function):
    """Computes the transposed convolution of two tensors over their last dimension."""

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
        if x.ndim != 3:
            raise ShapeError(f"Expected input to be 3D, got {x.ndim}D.")

        f = flip(f, -1)
        f = Dilation1DFn.forward(cache, f, dilation)
        x = Dilation1DFn.forward(cache, x, stride)
        x = Pad1DFn.forward(cache, x, f.shape[-1] - 1)  # full pad
        y = RawConv1DFn.forward(cache, x, f, stride=1)
        y = InvPad1DFn.forward(cache, y, padding)
        if b:
            y += b.view((*b.shape, 1))

        cache.push(b is not None)
        return y

    @staticmethod
    def backward(
        cache: FunctionCache, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        (b,) = cache.pop()

        dy = InvPad1DFn.backward(cache, dy)
        dx, df = RawConv1DFn.backward(cache, dy)
        dx = Pad1DFn.backward(cache, dx)
        dx = Dilation1DFn.backward(cache, dx)
        df = Dilation1DFn.backward(cache, df)
        df = flip(df, -1).to_contiguous()
        db = None if not b else dy.sum((0, 2))

        return dx, df, db


def conv_transpose1d(
    x: Tensor,
    f: Tensor,
    b: Optional[Tensor] = None,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
) -> Tensor:
    """Computes the transposed convolution of two tensors over their last dimension.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    b : Tensor, optional
        Bias tensor. Defaults to ``None. If ``None``, no bias is added.
    padding : int, optional
        Number of rows and cols removed from the output tensor. Defaults to ``0``.
    stride : int, optional
        Stride used in the deconvolution operation. Defaults to ``1``.
    dilation : int, optional
        Dilation factor used for the filter. Defaults to ``1``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.ConvTranspose1D`
    """
    return ConvTranspose1DFn.forward(PseudoCache(), x, f, b, padding, stride, dilation)


class InvPad2DFn(Function):
    """Removes rows and cols from a tensor's last two dimensions."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, padding: int) -> Tensor:
        no_padding = padding == 0
        cache.push(no_padding, padding)
        if no_padding:
            return x
        return x[..., padding:-padding, padding:-padding].to_contiguous()

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        no_padding, padding = cache.pop()
        if no_padding:
            return dy
        widths = tuple([(0, 0)] * (dy.ndim - 2) + [(padding, padding)] * 2)
        return pad(dy, widths)


class ConvTranspose2DFn(Function):
    """Computes the transposed convolution of two tensors over their last two dimensions."""

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
        if x.ndim != 4:
            raise ShapeError(f"Expected input to be 4D, got {x.ndim}D.")

        f = flip(f, (-2, -1))
        f = Dilation2DFn.forward(cache, f, dilation)
        x = Dilation2DFn.forward(cache, x, stride)
        x = Pad2DFn.forward(cache, x, f.shape[-1] - 1)  # full pad
        y = RawConv2DFn.forward(cache, x, f, stride=1)
        y = InvPad2DFn.forward(cache, y, padding)
        if b:
            y += b.view((*b.shape, 1, 1))

        cache.push(b is not None)
        return y

    @staticmethod
    def backward(
        cache: FunctionCache, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        (b,) = cache.pop()

        dy = InvPad2DFn.backward(cache, dy)
        dx, df = RawConv2DFn.backward(cache, dy)
        dx = Pad2DFn.backward(cache, dx)
        dx = Dilation2DFn.backward(cache, dx)
        df = Dilation2DFn.backward(cache, df)
        df = flip(df, (-2, -1)).to_contiguous()
        db = None if not b else dy.sum((0, 2, 3))

        return dx, df, db


def conv_transpose2d(
    x: Tensor,
    f: Tensor,
    b: Optional[Tensor] = None,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
) -> Tensor:
    """Computes the transposed convolution of two tensors over their last two dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    b : Tensor, optional
        Bias tensor. Defaults to ``None. If ``None``, no bias is added.
    padding : int, optional
        Number of rows and cols removed from the output tensor. Defaults to ``0``.
    stride : int, optional
        Stride used in the deconvolution operation. Defaults to ``1``.
    dilation : int, optional
        Dilation factor used for the filter. Defaults to ``1``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.Deconvolution2D`
    """
    return ConvTranspose2DFn.forward(PseudoCache(), x, f, b, padding, stride, dilation)
