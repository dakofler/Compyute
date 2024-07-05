"""Neural network functions module"""

from typing import Callable, Literal, Optional

from ...base_tensor import Tensor
from ...tensor_functions.creating import zeros
from ...tensor_functions.reshaping import (
    broadcast_to,
    flip,
    insert_dim,
    pad,
    pad_to_shape,
    repeat,
    reshape,
)
from ...tensor_functions.transforming import fft1d, fft2d, ifft1d, ifft2d
from ...tensor_functions.transforming import max as _max
from ...tensor_functions.transforming import mean, real
from ...tensor_functions.transforming import sum as _sum
from ...types import _AxisLike, _ShapeLike

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


def convolve1d(
    x: Tensor,
    f: Tensor,
    b: Optional[Tensor] = None,
    padding: Literal["valid", "same"] = "valid",
    stride: int = 1,
    dilation: int = 1,
    return_grad_func: bool = False,
) -> tuple[Tensor, Optional[Callable[[Tensor], tuple[Tensor, Optional[Tensor], Optional[Tensor]]]]]:
    """Convolves two tensors over their last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    b : Tensor, optional
        Bias tensor, by default None
    padding: Literal["valid", "same"], optional
        Padding applied to the input tensor, by default "valid".
    stride : int, optional
        Stride used in the convolution operation, by default 1.
    dilation : int, optional
        Dilation used for each axis of the filter, by default 1.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.
    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], tuple[Tensor, Optional[Tensor], Optional[Tensor]]]], optional
        Gradient function.
    """
    f, dil_grad_func = dilate1d(f, dilation, return_grad_func)  # (Co, Ci, F)
    f_ = reshape(f, (1,) + f.shape)  # (1, Co, Ci, F)
    f_.requires_grad = f.requires_grad

    p = _pad1d_from_str(padding, f_.shape[-1])
    x, pad_grad_func = pad1d(x, p, return_grad_func)  # (B, Ci, T)
    x_ = insert_dim(x, 1)  # (B, 1, Ci, T)

    conv, conv_grad_func = _convolve1d(x_, f_, stride, return_grad_func)  # (B, Co, Ci, T)
    y = _sum(conv, 2)  # (B, Co, T)

    if b is not None:
        y += reshape(b, (b.shape[0], 1))

    if return_grad_func:

        def grad_func(dy: Tensor) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
            dy_ = broadcast_to(insert_dim(dy, 2), conv.shape)
            dx, df = conv_grad_func(dy_)
            dx = pad_grad_func(_sum(dx, 1))

            if df is not None:
                df = dil_grad_func(_sum(df, 0))

            if b is not None and b.requires_grad:
                db = _sum(dy, (0, 2))
            else:
                db = None

            return dx, df, db

        return y, grad_func

    return y, None


def dilate1d(
    x: Tensor, dilation: int, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Dilates a tensor in its last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dilation : int
        Dilation used.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.
    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    if dilation == 1:
        return x, (lambda dy: dy)

    dil_shape = (dilation * x.shape[-1] - 1,)
    x_dil = zeros(x.shape[:-1] + dil_shape, x.dtype, x.device)
    dil_slice = [slice(None)] * (x.ndim - 1) + [slice(None, None, dilation)]
    x_dil[*dil_slice] = x

    if return_grad_func:
        return x_dil, (lambda dy: dy[*dil_slice])

    return x_dil, None


def _pad1d_from_str(padding: Literal["valid", "same"], kernel_size: int) -> tuple[int, int]:
    if padding == "valid":
        return (0, 0)
    p = kernel_size // 2
    return (p, p)


def pad1d(
    x: Tensor, padding: tuple[int, int], return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Pads a tensor in its last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    padding : tuple[int, int]
        Padding width applied to the front and back of the last axis.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.
    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    if padding == (0, 0):
        return x, (lambda dy: dy)

    widths = tuple([(0, 0)] * (x.ndim - 1) + [padding])
    y = pad(x, widths)

    if return_grad_func:
        pad_grad_slice = [slice(None)] * (x.ndim - 1) + [slice(padding[0], -padding[0])]
        return y, (lambda dy: dy[*pad_grad_slice])

    return y, None


def _convolve1d(
    x: Tensor,
    f: Tensor,
    stride: int = 1,
    return_grad_func: bool = False,
) -> tuple[Tensor, Optional[Callable[[Tensor], tuple[Tensor, Optional[Tensor]]]]]:
    f_ = flip(f, -1)
    conv = _fft_conv1d(x, f_)
    stride_slice = [slice(None)] * (x.ndim - 1) + [slice(None, None, stride)]
    y = conv[*stride_slice]

    if return_grad_func:

        def grad_func(dy: Tensor) -> tuple[Tensor, Optional[Tensor]]:
            # fill elements skipped by strides with zeros
            dy_, _ = dilate1d(dy, stride)
            dy_ = pad_to_shape(dy_, conv.shape)

            # full pad dy
            dy_, _ = pad1d(dy_, (f.shape[-1] - 1, f.shape[-1] - 1))  # full pad dy
            dx = _fft_conv1d(dy_, f)

            dy_ = flip(dy_, axis=-1)
            df = _fft_conv1d(dy_, x) if f.requires_grad else None

            return dx, df

        return y, grad_func

    return y, None


def _fft_conv1d(x: Tensor, f: Tensor) -> Tensor:
    cdtype = "complex64"
    conv = real(
        ifft1d(fft1d(x, dtype=cdtype) * fft1d(f, n=x.shape[-1], dtype=cdtype), dtype=cdtype),
        dtype=x.dtype,
    )
    out = x.shape[-1] - f.shape[-1] + 1
    out_slice = [slice(None)] * (x.ndim - 1) + [slice(-out, None)]
    return conv[*out_slice]


def convolve2d(
    x: Tensor,
    f: Tensor,
    b: Optional[Tensor] = None,
    padding: Literal["valid", "same"] = "valid",
    stride: int = 1,
    dilation: int = 1,
    return_grad_func: bool = False,
) -> tuple[Tensor, Optional[Callable[[Tensor], tuple[Tensor, Optional[Tensor], Optional[Tensor]]]]]:
    """Convolves two tensors over their last two axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    b : Tensor, optional
        Bias tensor, by default None
    padding: Literal["valid", "same"], optional
        Padding applied to the input tensor, by default "valid".
    stride : int, optional
        Stride used in the convolution operation, by default 1.
    dilation : int, optional
        Dilation used for each axis of the filter, by default 1.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], tuple[Tensor, Optional[Tensor], Optional[Tensor]]]], optional
        Gradient function.
    """
    d = (dilation, dilation)
    f, dil_grad_func = dilate2d(f, d, return_grad_func)  # (Co, Ci, Fy, Fx)
    f_ = reshape(f, (1,) + f.shape)  # (1, Co, Ci, Fy, Fx)
    f_.requires_grad = f.requires_grad

    p = _pad2d_from_str(padding, f_.shape[-1])
    x, pad_grad_func = pad2d(x, p, return_grad_func)  # (B, Ci, Y, X)
    x_ = insert_dim(x, 1)  # (B, 1, Ci, Y, X)

    s = (stride, stride)
    conv, conv_grad_func = _convolve2d(x_, f_, s, return_grad_func)  # (B, Co, Ci, Y, X)
    y = _sum(conv, 2)  # (B, Co, Y, X) sum over in channels

    if b is not None:
        y += reshape(b, (b.shape[0], 1, 1))

    if return_grad_func:

        def grad_func(dy: Tensor) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
            dy_ = broadcast_to(insert_dim(dy, 2), conv.shape)
            dx, df = conv_grad_func(dy_)
            dx = pad_grad_func(_sum(dx, 1))  # sum over out channels

            if df is not None:
                df = dil_grad_func(_sum(df, 0))  # sum over batches

            if b is not None and b.requires_grad:
                db = _sum(dy, (0, 2, 3))  # sum over batches, Y and X
            else:
                db = None

            return dx, df, db

        return y, grad_func

    return y, None


def dilate2d(
    x: Tensor, dilation: tuple[int, int], return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Dilates a tensor in its last two axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dilation : tuple[int, int]
        Dilation used.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.
    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
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
        [slice(None)] * (x.ndim - 2)
        + [slice(None, None, dilation[0])]
        + [slice(None, None, dilation[1])]
    )
    x_dil[*dil_slice] = x

    if return_grad_func:
        return x_dil, (lambda dy: dy[*dil_slice])

    return x_dil, None


def _pad2d_from_str(
    padding: Literal["valid", "same"], kernel_size: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if padding == "valid":
        return ((0, 0), (0, 0))
    p = kernel_size // 2
    return ((p, p), (p, p))


def pad2d(
    x: Tensor, padding: tuple[tuple[int, int], tuple[int, int]], return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Pads a tensor in its last two axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    padding : tuple[tuple[int, int], tuple[int, int]]
        Padding width applied to the front and back of the last two axes.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.
    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    if padding == ((0, 0), (0, 0)):
        return x, (lambda dy: dy)

    widths = tuple([(0, 0)] * (x.ndim - 2) + [*padding])
    y = pad(x, widths)

    if return_grad_func:
        pad_grad_slice = [slice(None)] * (x.ndim - 2) + [
            slice(padding[0][0], -padding[0][1]),
            slice(padding[1][0], -padding[1][1]),
        ]
        return y, (lambda dy: dy[*pad_grad_slice])

    return y, None


def _convolve2d(
    x: Tensor, f: Tensor, strides: tuple[int, int] = (1, 1), return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], tuple[Tensor, Optional[Tensor]]]]]:
    f_ = flip(f, (-2, -1))
    conv = _fft_conv2d(x, f_)
    stride_slice = [slice(None)] * (x.ndim - 2) + [
        slice(None, None, strides[0]),
        slice(None, None, strides[1]),
    ]
    y = conv[*stride_slice]

    if return_grad_func:

        def grad_func(dy: Tensor) -> tuple[Tensor, Optional[Tensor]]:
            # fill elements skipped by strides with zeros
            dy_, _ = dilate2d(dy, strides)
            dy_ = pad_to_shape(dy_, conv.shape)

            # full pad dy
            dy_, _ = pad2d(
                dy_, ((f.shape[-2] - 1, f.shape[-2] - 1), (f.shape[-1] - 1, f.shape[-1] - 1))
            )
            dx = _fft_conv2d(dy_, f)

            dy_ = flip(dy_, axis=(-2, -1))
            df = _fft_conv2d(dy_, x) if f.requires_grad else None

            return dx, df

        return y, grad_func

    return y, None


def _fft_conv2d(x: Tensor, f: Tensor) -> Tensor:
    cdtype = "complex64"
    conv = real(
        ifft2d(fft2d(x, dtype=cdtype) * fft2d(f, s=x.shape[-2:], dtype=cdtype), dtype=cdtype),
        dtype=x.dtype,
    )
    out_y = x.shape[-2] - f.shape[-2] + 1
    out_x = x.shape[-1] - f.shape[-1] + 1
    out_slice = [slice(None)] * (x.ndim - 2) + [
        slice(-out_y, None),
        slice(-out_x, None),
    ]
    return conv[*out_slice]


def upsample2d(
    x: Tensor,
    scaling_factors: tuple[int, int],
    shape: _ShapeLike,
    axes: _AxisLike = (-2, -1),
) -> Tensor:
    """Upsamples a tensor by repeating it's elements over given axes.

    Parameters
    ----------
    x : Tensor
        Tensor to be stretched out.
    scaling_factors : tuple[int, int]
        Number of repeating values along each axis.
    shape : ShapeLike
        Shape of the target tensor. If the shape does not match after stretching,
        remaining values are filled with zeroes.
    axes : AxisLike, optional
        Axes along which to stretch the tensor, by default (-2, -1).

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
    x: Tensor, kernel_size: tuple[int, int] = (2, 2), return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Performs a max pooling over the last two axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    kernel_size : tuple[int, int], optional
        Size of the pooling window, by default (2, 2).
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    Yi, Xi = x.shape[-2:]
    Ky, Kx = kernel_size

    # maxpooling
    crop_slice = [slice(None)] * (x.ndim - 2) + [
        slice(None, Yi // Ky * Ky),
        slice(None, Xi // Kx * Kx),
    ]
    x_crop = x[*crop_slice]
    pool_shape = x.shape[:-2] + (Yi // Ky, Ky, Xi // Kx, Kx)
    y = _max(reshape(x_crop, pool_shape), axis=(-3, -1))

    if return_grad_func:
        y_ups = upsample2d(y, kernel_size, x.shape)

        def grad_func(dy: Tensor) -> Tensor:
            return upsample2d(dy, kernel_size, x.shape) * (x == y_ups)

        return y, grad_func

    return y, None


def avgpooling2d(
    x: Tensor, kernel_size: tuple[int, int] = (2, 2), return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Performs a average pooling over the last two axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    kernel_size : tuple[int, int], optional
        Size of the pooling window, by default (2, 2).
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    Yi, Xi = x.shape[-2:]
    Ky, Kx = kernel_size

    # avgpooling
    crop_slice = [slice(None)] * (x.ndim - 2) + [
        slice(None, Yi // Ky * Ky),
        slice(None, Xi // Kx * Kx),
    ]
    x_crop = x[*crop_slice]
    pool_shape = x.shape[:-2] + (Yi // Ky, Ky, Xi // Kx, Kx)
    y = mean(reshape(x_crop, pool_shape), axis=(-3, -1))

    if return_grad_func:

        def grad_func(dy: Tensor) -> Tensor:
            return upsample2d(dy, kernel_size, x.shape) / (Ky * Kx)

        return y, grad_func

    return y, None
