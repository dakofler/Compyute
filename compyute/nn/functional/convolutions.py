"""Neural network convolution functions."""

from typing import Callable, Literal, Optional

from ...base_tensor import ShapeLike, Tensor
from ...tensor_ops.creating import zeros
from ...tensor_ops.reshaping import broadcast_to, flip, pad, pad_to_shape, repeat
from ...tensor_ops.transforming import fft1d, fft2d, ifft1d, ifft2d
from ...tensor_ops.transforming import max as cpmax
from ...tensor_ops.transforming import mean, real
from ...tensor_ops.transforming import sum as cpsum

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

_PaddingLike = Literal["valid", "same"]


def convolve1d(
    x: Tensor,
    f: Tensor,
    b: Optional[Tensor] = None,
    padding: _PaddingLike = "valid",
    stride: int = 1,
    dilation: int = 1,
    return_grad_fn: bool = False,
) -> tuple[Tensor, Optional[Callable]]:
    """Computes the convolution of two tensors over their last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    b : Tensor, optional
        Bias tensor. Defaults to ``None``. If ``None``, no bias is added.
    padding : _PaddingLike, optional
        Padding applied to the input tensor. Defaults to ``valid``.
    stride : int, optional
        Stride used in the convolution operation. Defaults to ``1``.
    dilation : int, optional
        Dilation factor to use for each axis of the filter. Defaults to ``1``.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable, optional
        Gradient function.

    See Also
    ----------
    :class:`compyute.nn.Convolution1d`
    """
    # dilate filter and add a fake batch dimension
    f, dil_grad_fn = dilate1d(f, dilation, return_grad_fn)  # (Co, Ci, F)
    f_ext = f.to_shape((1, *f.shape))  # (1, Co, Ci, F)

    # pad input and add a fake output dimension
    p = pad1d_from_str(padding, f_ext.shape[-1])
    x, pad_grad_fn = pad1d(x, p, return_grad_fn)  # (B, Ci, T)
    x_ext = x.to_shape((x.shape[0], 1, *x.shape[1:]))  # (B, 1, Ci, T)

    # perform convolution and sum over input dimension
    # (B, Co, Ci, T)
    conv, conv_grad_fn = convolve1d_(x_ext, f_ext, stride, return_grad_fn)
    y = cpsum(conv, axis=2)  # (B, Co, T)

    if b:
        y += b.to_shape((*b.shape, 1))

    if conv_grad_fn is not None and pad_grad_fn is not None and dil_grad_fn is not None:

        def grad_fn(dy: Tensor) -> tuple[Tensor, Tensor, Optional[Tensor]]:
            # insert fake input channel dimension
            dy_ext = dy.to_shape((*dy.shape[:2], 1, *dy.shape[2:]))
            dx, df = conv_grad_fn(broadcast_to(dy_ext, conv.shape))

            dx = pad_grad_fn(cpsum(dx, axis=1))
            df = dil_grad_fn(cpsum(df, axis=0))
            db = cpsum(dy, axis=(0, 2)) if b else None

            return dx, df, db

        return y, grad_fn

    return y, None


def dilate1d(
    x: Tensor, dilation: int, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable]]:
    """Dilates a tensor in its last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dilation : int
        Dilation factor to use.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable, optional
        Gradient function.
    """
    if dilation == 1:
        return x, (lambda dy: dy)

    dil_shape = (dilation * x.shape[-1] - 1,)
    x_dil = zeros(x.shape[:-1] + dil_shape, x.dtype, x.device)
    dil_slice = [slice(None)] * (x.n_axes - 1) + [slice(None, None, dilation)]
    x_dil[*dil_slice] = x

    if return_grad_fn:
        return x_dil, (lambda dy: dy[*dil_slice])

    return x_dil, None


def pad1d_from_str(padding: _PaddingLike, kernel_size: int) -> tuple[int, int]:
    """Returns padding widths from a string."""
    if padding == "valid":
        return (0, 0)
    p = kernel_size // 2
    return (p, p)


def pad1d(
    x: Tensor, padding: tuple[int, int], return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable]]:
    """Pads a tensor in its last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    padding : tuple[int, int]
        Padding width applied to the beginning and end of the last axis.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable, optional
        Gradient function.
    """
    if padding == (0, 0):
        return x, (lambda dy: dy)

    widths = tuple([(0, 0)] * (x.n_axes - 1) + [padding])
    y = pad(x, widths)

    if return_grad_fn:
        pad_grad_slice = [slice(None)] * (x.n_axes - 1) + [
            slice(padding[0], -padding[0])
        ]
        return y, (lambda dy: dy[*pad_grad_slice])

    return y, None


def convolve1d_(
    x: Tensor, f: Tensor, stride: int = 1, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable]]:
    """Computes the 1D convolution of two tensors."""
    f_flipped = flip(f, -1)
    conv = fft_conv1d(x, f_flipped)
    stride_slice = [slice(None)] * (x.n_axes - 1) + [slice(None, None, stride)]
    y = conv[*stride_slice]

    if return_grad_fn:

        def grad_fn(dy: Tensor) -> tuple[Tensor, Tensor]:
            # fill elements skipped by strides with zeros
            dy, _ = dilate1d(dy, stride)
            dy = pad_to_shape(dy, conv.shape)

            dy, _ = pad1d(dy, (f.shape[-1] - 1, f.shape[-1] - 1))  # full pad dy
            dx = fft_conv1d(dy, f)

            dy = flip(dy, axis=-1)
            df = fft_conv1d(dy, x)

            return dx, df

        return y, grad_fn

    return y, None


def fft_conv1d(x: Tensor, f: Tensor) -> Tensor:
    """Computes the 1D convolution of two tensors using FFT."""
    conv = real(ifft1d(fft1d(x) * fft1d(f, n=x.shape[-1])), dtype=x.dtype)
    out = x.shape[-1] - f.shape[-1] + 1
    out_slice = [slice(None)] * (x.n_axes - 1) + [slice(-out, None)]
    return conv[*out_slice]


def convolve2d(
    x: Tensor,
    f: Tensor,
    b: Optional[Tensor] = None,
    padding: _PaddingLike = "valid",
    stride: int = 1,
    dilation: int = 1,
    return_grad_fn: bool = False,
) -> tuple[Tensor, Optional[Callable]]:
    """Computes the convolution of two tensors over their last two axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    b : Tensor, optional
        Bias tensor. Defaults to ``None. If ``None``, no bias is added.
    padding : _PaddingLike, optional
        Padding applied to the input tensor. Defaults to ``valid``.
    stride : int, optional
        Stride used in the convolution operation. Defaults to ``1``.
    dilation : int, optional
        Dilation factor to use for each axis of the filter. Defaults to ``1``.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable, optional
        Gradient function.

    See Also
    ----------
    :class:`compyute.nn.Convolution2d`
    """

    # dilate filter and add a fake batch dimension
    f, dil_grad_fn = dilate2d(f, (dilation, dilation), return_grad_fn)
    f_ext = f.to_shape((1, *f.shape))  # (1, Co, Ci, Fy, Fx)

    # pad input and add a fake output dimension
    p = pad2d_from_str(padding, f_ext.shape[-1])
    x, pad_grad_fn = pad2d(x, p, return_grad_fn)
    x_ext = x.to_shape((x.shape[0], 1, *x.shape[1:]))  # (B, 1, Ci, Y, X)

    # perform convolution and sum over input dimension
    # (B, Co, Ci, Y, X)
    conv, conv_grad_fn = convolve2d_(x_ext, f_ext, (stride, stride), return_grad_fn)
    y = cpsum(conv, axis=2)  # (B, Co, Y, X)

    if b:
        y += b.to_shape((*b.shape, 1, 1))

    if conv_grad_fn is not None and pad_grad_fn is not None and dil_grad_fn is not None:

        def grad_fn(dy: Tensor) -> tuple[Tensor, Tensor, Optional[Tensor]]:
            # insert fake input channel dimension
            dy_ext = dy.to_shape((*dy.shape[:2], 1, *dy.shape[2:]))
            dx, df = conv_grad_fn(broadcast_to(dy_ext, conv.shape))

            dx = pad_grad_fn(cpsum(dx, axis=1))  # sum over out channels
            df = dil_grad_fn(cpsum(df, axis=0))  # sum over batches
            db = cpsum(dy, axis=(0, 2, 3)) if b else None

            return dx, df, db

        return y, grad_fn

    return y, None


def dilate2d(
    x: Tensor, dilation: tuple[int, int], return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable]]:
    """Dilates a tensor in its last two axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dilation : tuple[int, int]
        Dilation factor to use.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable, optional
        Gradient function.
    """
    if dilation == (1, 1):
        return x, (lambda dy: dy)

    dil_shape = (
        dilation[0] * x.shape[-2] - 1,
        dilation[1] * x.shape[-1] - 1,
    )
    x_dil = zeros(x.shape[:-2] + dil_shape, x.dtype, x.device)
    dil_slice = (
        [slice(None)] * (x.n_axes - 2)
        + [slice(None, None, dilation[0])]
        + [slice(None, None, dilation[1])]
    )
    x_dil[*dil_slice] = x

    if return_grad_fn:
        return x_dil, (lambda dy: dy[*dil_slice])

    return x_dil, None


def pad2d_from_str(
    padding: _PaddingLike, kernel_size: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Returns padding widths from a string."""
    if padding == "valid":
        return ((0, 0), (0, 0))
    p = kernel_size // 2
    return ((p, p), (p, p))


def pad2d(
    x: Tensor,
    padding: tuple[tuple[int, int], tuple[int, int]],
    return_grad_fn: bool = False,
) -> tuple[Tensor, Optional[Callable]]:
    """Pads a tensor in its last two axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    padding : tuple[tuple[int, int], tuple[int, int]]
        Padding width applied to the beginning and end of the last two axes.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable, optional
        Gradient function.
    """
    if padding == ((0, 0), (0, 0)):
        return x, (lambda dy: dy)

    widths = tuple([(0, 0)] * (x.n_axes - 2) + [*padding])
    y = pad(x, widths)

    if return_grad_fn:
        pad_grad_slice = [slice(None)] * (x.n_axes - 2) + [
            slice(padding[0][0], -padding[0][1]),
            slice(padding[1][0], -padding[1][1]),
        ]
        return y, (lambda dy: dy[*pad_grad_slice])

    return y, None


def convolve2d_(
    x: Tensor,
    f: Tensor,
    strides: tuple[int, int] = (1, 1),
    return_grad_fn: bool = False,
) -> tuple[Tensor, Optional[Callable]]:
    """Computes the 2D convolution of two tensors."""
    f_flipped = flip(f, (-2, -1))
    conv = fft_conv2d(x, f_flipped)
    stride_slice = [slice(None)] * (x.n_axes - 2) + [
        slice(None, None, strides[0]),
        slice(None, None, strides[1]),
    ]
    y = conv[*stride_slice]

    if return_grad_fn:

        def grad_fn(dy: Tensor) -> tuple[Tensor, Tensor]:
            # fill elements skipped by strides with zeros
            dy, _ = dilate2d(dy, strides)
            dy = pad_to_shape(dy, conv.shape)

            # full pad dy
            dy, _ = pad2d(
                dy,
                (
                    (f.shape[-2] - 1, f.shape[-2] - 1),
                    (f.shape[-1] - 1, f.shape[-1] - 1),
                ),
            )
            dx = fft_conv2d(dy, f)

            dy = flip(dy, axis=(-2, -1))
            df = fft_conv2d(dy, x)

            return dx, df

        return y, grad_fn

    return y, None


def fft_conv2d(x: Tensor, f: Tensor) -> Tensor:
    """Computes the 2D convolution of two tensors using FFT."""
    conv = real(ifft2d(fft2d(x) * fft2d(f, s=x.shape[-2:])), dtype=x.dtype)
    out_y = x.shape[-2] - f.shape[-2] + 1
    out_x = x.shape[-1] - f.shape[-1] + 1
    out_slice = [slice(None)] * (x.n_axes - 2) + [
        slice(-out_y, None),
        slice(-out_x, None),
    ]
    return conv[*out_slice]


def upsample2d(
    x: Tensor,
    scaling_factors: tuple[int, int],
    shape: ShapeLike,
    axes: tuple[int, int] = (-2, -1),
) -> Tensor:
    """Upsamples a tensor by repeating it's elements over given axes.

    Parameters
    ----------
    x : Tensor
        Tensor to be stretched out.
    scaling_factors : tuple[int, int]
        Number of repeating values along each axis.
    shape : ShapeLike
        Shape of the target tensor. If the shape does not match after upsampling,
        remaining values are filled with zeroes.
    axes : tuple[int, int], optional
        Axes along which to stretch the tensor. Defaults to ``(-2, -1)``.

    Returns
    -------
    Tensor
        Upsampled tensor.
    """
    sf1, sf2 = scaling_factors
    ax1, ax2 = axes
    x_str = repeat(repeat(x, sf1, ax1), sf2, ax2)
    return x_str if x_str.shape == shape else pad_to_shape(x_str, shape)


def maxpooling2d(
    x: Tensor, kernel_size: tuple[int, int] = (2, 2), return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable]]:
    """Performs max pooling over the last two axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    kernel_size : tuple[int, int], optional
        Size of the pooling window. Defaults to ``(2, 2)``.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable, optional
        Gradient function.

    See Also
    ----------
    :class:`compyute.nn.MaxPooling2D`
    """
    x_height, x_width = x.shape[-2:]
    kernel_height, kernel_width = kernel_size

    # maxpooling
    crop_slice = [slice(None)] * (x.n_axes - 2) + [
        slice(None, x_height // kernel_height * kernel_height),
        slice(None, x_width // kernel_width * kernel_width),
    ]
    x_crop = x[*crop_slice]
    pool_shape = x.shape[:-2] + (
        x_height // kernel_height,
        kernel_height,
        x_width // kernel_width,
        kernel_width,
    )
    y = cpmax(x_crop.to_shape(pool_shape), axis=(-3, -1))

    if return_grad_fn:
        y_ups = upsample2d(y, kernel_size, x.shape)
        return y, lambda dy: upsample2d(dy, kernel_size, x.shape) * (x == y_ups)

    return y, None


def avgpooling2d(
    x: Tensor, kernel_size: tuple[int, int] = (2, 2), return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable]]:
    """Performs average pooling over the last two axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    kernel_size : tuple[int, int], optional
        Size of the pooling window. Defaults to ``(2, 2)``.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable, optional
        Gradient function.

    See Also
    ----------
    :class:`compyute.nn.AvgPooling2D`
    """
    x_height, x_width = x.shape[-2:]
    kernel_height, kernel_width = kernel_size

    # avgpooling
    crop_slice = [slice(None)] * (x.n_axes - 2) + [
        slice(None, x_height // kernel_height * kernel_height),
        slice(None, x_width // kernel_width * kernel_width),
    ]
    x_crop = x[*crop_slice]
    pool_shape = x.shape[:-2] + (
        x_height // kernel_height,
        kernel_height,
        x_width // kernel_width,
        kernel_width,
    )
    y = mean(x_crop.to_shape(pool_shape), axis=(-3, -1))

    if return_grad_fn:

        def grad_fn(dy: Tensor) -> Tensor:
            return upsample2d(dy, kernel_size, x.shape) / (kernel_height * kernel_width)

        return y, grad_fn

    return y, None
