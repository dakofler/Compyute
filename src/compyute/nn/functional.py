"""Neural network functions module"""

from typing import Callable, Literal, Optional

from .._tensor import Tensor
from .._tensor_functions._computing import maximum, minimum, tensorprod
from .._tensor_functions._creating import identity, zeros
from .._tensor_functions._reshaping import (
    broadcast_to,
    flip,
    insert_dim,
    pad,
    pad_to_shape,
    repeat,
    reshape,
    tile,
)
from .._tensor_functions._selecting import argmax
from .._tensor_functions._transforming import clip, exp, fft1d, fft2d, ifft1d, ifft2d, log
from .._tensor_functions._transforming import max as _max
from .._tensor_functions._transforming import mean, real, sech
from .._tensor_functions._transforming import sum as _sum
from .._tensor_functions._transforming import tanh as _tanh
from .._types import _AxisLike, _ShapeLike
from ..preprocessing._basic import one_hot_encode

__all__ = [
    "relu",
    "leaky_relu",
    "gelu",
    "sigmoid",
    "tanh",
    "softmax",
    "temperature_softmax",
    "linear",
    "convolve1d",
    "dilate1d",
    "pad1d",
    "convolve2d",
    "dilate2d",
    "pad2d",
    "upsample2d",
    "maxpooling2d",
    "avgpooling2d",
    "lookup_embedding",
    "mean_squared_error",
    "cross_entropy",
    "binary_cross_entropy",
    "accuracy_score",
    "r2_score",
]
PI: float = 3.141592653589793
GELU_S: float = 0.7978845608028654  # sqrt(2/pi)
GELU_C: float = 0.044715


# ----------------------------------------------------------------------------------------------
# ACTIVATION FUNCTIONS
# ----------------------------------------------------------------------------------------------


def relu(
    x: Tensor, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the Rectified Linear Unit function.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    y = maximum(x, 0)

    if return_grad_func:
        return y, (lambda dy: (y > 0) * dy)
    return y, None


def leaky_relu(
    x: Tensor, alpha: float = 0.01, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the leaky ReLU function.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    alpha : float, optional
        Slope of the negative output, by default 0.01.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    x = x.float()
    y = maximum(x, 0) + alpha * minimum(0, x)

    if return_grad_func:
        return y, (lambda dy: ((y > 0).float() + (y < 0).float() * alpha) * dy)
    return y, None


def gelu(
    x: Tensor, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the Gaussian Error Linear Unit function.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """

    tmp = GELU_S * (x + GELU_C * x**3)
    y = 0.5 * x * (1 + _tanh(tmp))

    if return_grad_func:
        return y, (
            lambda dy: (
                0.5 * (1 + _tanh(tmp)) + 0.5 * x * sech(tmp) ** 2 * GELU_S * (1 + 3 * GELU_C * x**2)
            )
            * dy
        )
    return y, None


def sigmoid(
    x: Tensor, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the sigmoid function.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    y = exp(x) * (1 + exp(x)) ** -1

    if return_grad_func:
        return y, (lambda dy: (y * (1 - y)) * dy)
    return y, None


def tanh(
    x: Tensor, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the hyperbolic tangent function.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    y = _tanh(x)

    if return_grad_func:
        return y, (lambda dy: (1 - y**2) * dy)
    return y, None


def softmax(
    x: Tensor, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the softmax function over the last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    x = exp(x - _max(x, axis=-1, keepdims=True))
    y = x / _sum(x, axis=-1, keepdims=True)

    if return_grad_func:

        def grad_func(dy: Tensor) -> Tensor:
            sm_ = tile(insert_dim(y, -1), y.shape[-1], -1)
            return reshape(
                sm_ * (identity(y.shape[-1], device=x.device) - sm_.T) @ insert_dim(dy, -1), y.shape
            )

        return y, grad_func
    return y, None


def temperature_softmax(x: Tensor, temperature: float = 1) -> Tensor:
    """Applies the softmax function with temperature to the last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    temperature : float, optional
        Temperature scaling to be applied in the calculation.

    Returns
    -------
    Tensor
        Output tensor.
    """
    if temperature == 0:
        raise ValueError("Temperature cannot be 0.")

    x = exp((x - _max(x, axis=-1, keepdims=True)) / temperature)
    return x / _sum(x, axis=-1, keepdims=True)


# ----------------------------------------------------------------------------------------------
# TRAINABLE LAYER FUNCTIONS
# ----------------------------------------------------------------------------------------------


def linear(
    x: Tensor, w: Tensor, b: Optional[Tensor] = None, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], tuple[Tensor, Optional[Tensor], Optional[Tensor]]]]]:
    """Applies the linear transformation X @ W^T + b.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w : Tensor
        Weight tensor.
    b : Tensor, optional
        Bias tensor, by default None
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Linearly transformed tensor.
    Callable[[Tensor], tuple[Tensor, Tensor, Optional[Tensor]]], optional
        Gradient function.
    """

    y = x @ w.T
    if b is not None:
        y += b

    if return_grad_func:

        def grad_func(dy: Tensor) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
            # input grads
            dx = dy @ w

            # weight grads
            if w.requires_grad:
                dw = dy.T @ x
                if x.ndim > 2:  # sum over all batch dimensions
                    axes = tuple(range(x.ndim - 2))
                    dw = _sum(dw, axis=axes)
            else:
                dw = None

            # bias grads
            if b is not None and b.requires_grad:
                axes = tuple(range(x.ndim - 1))
                db = _sum(dy, axis=axes)  # sum over all batch dimensions
            else:
                db = None

            return dx, dw, db

        return y, grad_func

    return y, None


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
    B, C, Yi, Xi = x.shape
    Ky, Kx = kernel_size

    # maxpooling
    x_crop = x[:, :, : Yi // Ky * Ky, : Xi // Kx * Kx]
    y = _max(reshape(x_crop, (B, C, Yi // Ky, Ky, Xi // Kx, Kx)), axis=(-3, -1))

    if return_grad_func:
        y_ups = upsample2d(y, kernel_size, x.shape)
        grad_func = (
            (lambda dy: upsample2d(dy, kernel_size, x.shape) * (x == y_ups).int())
            if return_grad_func
            else None
        )
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
    B, C, Yi, Xi = x.shape
    Ky, Kx = kernel_size

    # avgpooling
    x_crop = x[:, :, : Yi // Ky * Ky, : Xi // Kx * Kx]
    y = mean(reshape(x_crop, (B, C, Yi // Ky, Ky, Xi // Kx, Kx)), axis=(-3, -1))

    if return_grad_func:
        grad_func = (
            (lambda dy: upsample2d(dy, kernel_size, x.shape) / (Ky * Kx))
            if return_grad_func
            else None
        )
        return y, grad_func

    return y, None


def lookup_embedding(
    x: Tensor, embedding_table: Tensor, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Optional[Tensor]]]]:
    """Performs lookup embedding on a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor of integer dtype used for indexing into the embedding table.
    embedding_table : Tensor
        Tensor of embedding values.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    if x.dtype not in ("int32", "int64"):
        raise ValueError(f"Input must be int32 or int64, got {x.dtype}.")

    x = one_hot_encode(x, embedding_table.shape[0]).astype(embedding_table.dtype)
    y = x @ embedding_table

    if return_grad_func:

        def grad_func(dy: Tensor) -> Optional[Tensor]:
            # embedding table grads
            if embedding_table.requires_grad:
                return _sum(x.T @ dy, axis=0)

        return y, grad_func

    return y, None


# ----------------------------------------------------------------------------------------------
# LOSS FUNCTIONS
# ----------------------------------------------------------------------------------------------


def mean_squared_error(
    y: Tensor, t: Tensor, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[], Tensor]]]:
    """Computes the mean squared error loss.

    Parameters
    ----------
    y : Tensor
        A model's predictions.
    t : Tensor
        Target values.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Mean squared error loss.
    Callable[[], Tensor]], optional
        Gradient function.
    """
    dif = y.float() - t.float()
    loss = mean(dif**2)

    grad_func = (lambda: dif * 2 / tensorprod(y.shape)) if return_grad_func else None

    return loss, grad_func


def cross_entropy(
    y: Tensor, t: Tensor, eps: float = 1e-8, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[], Tensor]]]:
    """Computes the cross entropy loss.

    Parameters
    ----------
    y : Tensor
        Model logits.
    t : Tensor
        Target integer class labels.
    eps : float, optional
        Constant used for numerical stability, by default 1e-8.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Cross entropy loss.
    Callable[[], Tensor]], optional
        Gradient function.
    """
    probs, _ = softmax(y.float(), False)
    t = one_hot_encode(t.int(), y.shape[-1])
    loss = -mean(log(_sum((probs + eps) * t, axis=-1)))

    grad_func = (lambda: (probs - t) / tensorprod(y.shape[:-1])) if return_grad_func else None

    return loss, grad_func


def binary_cross_entropy(
    y: Tensor, t: Tensor, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[], Tensor]]]:
    """Computes the cross entropy loss.

    Parameters
    ----------
    y : Tensor
        Model logits.
    t : Tensor
        Binary target values.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Cross entropy loss.
    Callable[[], Tensor]], optional
        Gradient function.
    """
    c = 100
    loss = -mean(t * clip(log(y), -c, c) + (1 - t) * clip(log(1 - y), -c, c))

    grad_func = (
        (lambda: (-t / y + (1 - t) / (1 - y)) / tensorprod(y.shape)) if return_grad_func else None
    )

    return loss, grad_func


# ----------------------------------------------------------------------------------------------
# METRIC FUNCTIONS
# ----------------------------------------------------------------------------------------------


def accuracy_score(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Computes the accuracy score.

    Parameters
    ----------
    y_pred : Tensor
        A model's predictions.
    y_true : Tensor
        Target values.

    Returns
    -------
    Tensor
        Accuracy score.
    """
    return _sum(argmax(y_pred, -1) == y_true) / tensorprod(y_pred.shape[:-1])


def r2_score(y_pred: Tensor, y_true: Tensor, eps: float = 1e-8) -> Tensor:
    """Computes the coefficient of determination (R2 score).

    Parameters
    ----------
    y_pred : Tensor
        A model's predictions.
    y_true : Tensor
        Target values.
    eps: float, optional
        Constant for numerical stability, by default 1e-8.

    Returns
    -------
    Tensor
        R2 score.
    """
    ssr = _sum((y_true - y_pred) ** 2)
    sst = _sum((y_true - mean(y_true)) ** 2)
    return 1 - ssr / (sst + eps)
