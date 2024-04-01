"""Neural network functions module"""

from typing import Literal
from ..tensor_f import arange, maximum, minimum, zeros
from ..tensor import Tensor, ShapeError
from ..types import ShapeLike


__all__ = [
    "relu",
    "leaky_relu",
    "gelu",
    "sigmoid",
    "softmax",
    "convolve1d",
    "convolve2d",
]
pi: float = 3.141592653589793


def relu(x: Tensor) -> Tensor:
    """Applies the Rectified Linear Unit function.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return maximum(x, 0)


def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    """Applies the leaky ReLU function.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    alpha : float
        Slope of the negative output.

    Returns
    -------
    Tensor
        Output tensor.
    """
    if "int" in str(x.dtype):
        x = x.float()
    return maximum(x, 0) + alpha * minimum(0, x)


def gelu(x: Tensor) -> Tensor:
    """Applies the Gaussian Error Linear Unit function.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return 0.5 * x * (1 + ((2 / pi) ** 0.5 * (x + 0.044715 * x**3)).tanh())


def sigmoid(x: Tensor) -> Tensor:
    """Applies the sigmoid function.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return x.exp() * (1 + x.exp()) ** -1


def softmax(x: Tensor) -> Tensor:
    """Applies the softmax function to the last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    x = (x - x.max(axis=-1, keepdims=True)).exp()
    return x / x.sum(axis=-1, keepdims=True)


def log_softmax(x: Tensor) -> Tensor:
    """Applies the log softmax function to the last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    x = (x - x.max(axis=-1, keepdims=True)).exp()
    return (x / x.sum(axis=-1, keepdims=True)).log()


def linear(x: Tensor, w: Tensor, b: Tensor | None = None) -> Tensor:
    """Applies the linear transformation X @ W^T + b.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w : Tensor
        Weight tensor.
    b : Tensor | None, optional
        Bias tensor, by default None

    Returns
    -------
    Tensor
        Linearly transformed tensor.
    """
    if b is None:
        return x @ w.T
    return x @ w.T + b


def linear_backward(
    dy: Tensor, x: Tensor, w: Tensor, b: Tensor | None = None
) -> Tensor:
    """Backpropagates through a linear transformation.

    Parameters
    ----------
    dy : Tensor
        Output gradients.
    x : Tensor
        Input tensor.
    w : Tensor
        Weight tensor.
    b : Tensor | None, optional
        Bias tensor, by default None

    Returns
    -------
    Tensor
        Input gradients.
    """

    # input grads
    # (B, ... , Co) @ (Co, Ci) -> (B, ..., Ci)
    dx = dy @ w

    # weight grads
    # 2D: (Co, B) @ (B, Ci) -> (Co, Ci)
    # ND: (B, ..., Co, Bn) @ (B, ... , Bn, Ci) -> (B, ..., Co, Ci)
    dw = dy.transpose() @ x
    if x.ndim > 2:
        # sum over all batch dimensions
        # (B, ..., Ci, Co) -> (Ci, Co)
        dw = dw.sum(axis=tuple(arange(x.ndim - 2)))

    w.grad = dw

    # bias grads
    if b is not None:
        # sum over all batch dimensions
        # (B, ... , Co) -> (Co,)
        b.grad = dy.sum(axis=tuple(arange(x.ndim - 1)))

    return dx


def convolve1d(
    x: Tensor,
    f: Tensor,
    padding: Literal["causal", "full", "same", "valid"] = "valid",
    stride: int = 1,
    dilation: int = 1,
) -> Tensor:
    """Convolves two tensors over their last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    padding: Literal["causal", "full", "same", "valid"], optional
        Padding applied to a tensor before the convolution, by default "causal".
    stride : int, optional
        Stride used in the convolution operation, by default 1.
    dilation : int, optional
        Dilation used for each axis of the filter, by default 1.
    Returns
    -------
    Tensor
        Output tensor.

    Raises
    -------
    ShapeError
        If dimensions of input and filter do not match.
    """
    if x.ndim != f.ndim:
        raise ShapeError("Dimensions of input and filter must match.")

    if dilation > 1:
        f = __dilate1d(f, dilation)

    if padding != "valid":
        p = __pad1d_from_str(padding, f.shape[-1])
        x = __pad1d(x, p)

    return __convolve1d(x, f, stride)


def __pad1d_from_str(
    padding: Literal["causal", "full", "same", "valid"], kernel_size: int
) -> tuple[int, int]:
    match padding:
        case "causal":
            return (kernel_size - 1, 0)
        case "full":
            return (kernel_size - 1,) * 2
        case "same":
            return (kernel_size // 2,) * 2
        case _:
            return (0, 0)


def __pad1d(x: Tensor, padding: tuple[int, int]) -> Tensor:
    widths = tuple([(0, 0)] * (x.ndim - 1) + [padding])
    return x.pad(widths)


def __dilate1d(x: Tensor, dilation: int) -> Tensor:
    if dilation == 1:
        return x

    dilated_shape = (dilation * x.shape[-1] - 1,)
    x_dilated = zeros(x.shape[:-1] + dilated_shape, x.dtype, x.device)
    dilation_slice = [slice(None)] * (x.ndim - 1) + [slice(None, None, dilation)]
    x_dilated[*dilation_slice] = x
    return x_dilated


def __convolve1d(x: Tensor, f: Tensor, stride: int = 1) -> Tensor:
    if x.ndim != f.ndim:
        raise ShapeError("Dimensions of input and filter must match.")

    # convolution
    cdtype = "complex64"
    conv = (
        (x.fft1d(dtype=cdtype) * f.fft1d(n=x.shape[-1], dtype=cdtype))
        .ifft1d(dtype=cdtype)
        .real(dtype=x.dtype)
    )

    # slicing
    out = 1 + (x.shape[-1] - f.shape[-1])
    out_slice = [slice(None)] * (x.ndim - 1) + [slice(-out, None)]
    stride_slice = [slice(None)] * (x.ndim - 1) + [slice(None, None, stride)]

    return conv[*out_slice][*stride_slice]


def convolve2d(
    x: Tensor,
    f: Tensor,
    padding: Literal["same", "valid"] = "valid",
    stride: int = 1,
    dilation: int = 1,
) -> Tensor:
    """Convolves two tensors over their last two axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    padding: Literal["same", "valid"], optional
        Padding applied to a tensor before the convolution, by default "valid".
    stride : int, optional
        Stride used in the convolution operation, by default 1.
    dilation : int, optional
        Dilation used for each axis of the filter, by default 1.

    Returns
    -------
    Tensor
        Output tensor.

    Raises
    -------
    ShapeError
        If dimensions of input and filter do not match.
    """
    if x.ndim != f.ndim:
        raise ShapeError("Dimensions of input and filter must match.")

    if dilation > 1:
        f = __dilate2d(f, (dilation, dilation))

    if padding != "valid":
        p = __pad2d_from_str(padding, f.shape[-1])
        x = __pad2d(x, p)

    return __convolve2d(x, f, (stride, stride))


def __pad2d_from_str(padding: Literal["same", "valid"], kernel_size: int):
    match padding:
        case "full":
            return ((kernel_size - 1,) * 2,) * 2
        case "same":
            return ((kernel_size // 2,) * 2,) * 2
        case _:
            return ((0, 0), (0, 0))


def __pad2d(x: Tensor, padding: tuple[tuple[int, int]]) -> Tensor:
    """Pads 2 trailing dimensions of a tensor (axes=(-2, -1)).

    Parameters
    ----------
    x : Tensor
        Tensor to be padded.
    padding : tuple[tuple[int, int]]
        Padding applied before and after the elements of each axis.

    Returns
    -------
    Tensor
        Padded tensor.
    """
    widths = tuple([(0, 0)] * (x.ndim - 2) + [*padding])
    return x.pad(widths)


def __dilate2d(x: Tensor, dilation: tuple[int, int]) -> Tensor:
    """Dilates a tensor over it's last two axes.

    Parameters
    ----------
    x : Tensor
        Tensor to be dilated.
    dilation : tuple [int, int]
        Dilation used for each axis of the tensor.

    Returns
    -------
    Tensor
        Dilated tensor.
    """
    dilated_shape = (
        dilation[0] * x.shape[-2] - 1,
        dilation[1] * x.shape[-1] - 1,
    )
    x_dilated = zeros(x.shape[:-2] + dilated_shape, x.dtype, x.device)
    dilation_slice = (
        [slice(None)] * (x.ndim - 2)
        + [slice(None, None, dilation[0])]
        + [slice(None, None, dilation[1])]
    )
    x_dilated[*dilation_slice] = x
    return x_dilated


def __convolve2d(
    x: Tensor,
    f: Tensor,
    strides: tuple[int, int] = (1, 1),
) -> Tensor:

    # convolution
    cdtype = "complex64"
    conv = (
        (x.fft2d(dtype=cdtype) * f.fft2d(s=x.shape[-2:], dtype=cdtype))
        .ifft2d(dtype=cdtype)
        .real(dtype=x.dtype)
    )

    # slicing
    out_y = 1 + (x.shape[-2] - f.shape[-2])
    out_x = 1 + (x.shape[-1] - f.shape[-1])
    out_slice = [slice(None)] * (x.ndim - 2) + [
        slice(-out_y, None),
        slice(-out_x, None),
    ]
    stride_slice = [slice(None)] * (x.ndim - 2) + [
        slice(None, None, strides[0]),
        slice(None, None, strides[1]),
    ]

    return conv[*out_slice][*stride_slice]


def stretch2d(
    x: Tensor,
    stretches: tuple[int, int],
    shape: ShapeLike,
    axes: tuple[int, int] = (-2, -1),
) -> Tensor:
    """Stretches a tensor by repeating it's elements over given axes.

    Parameters
    ----------
    x : Tensor
        Tensor to be stretched out.
    stretches : tuple[int, int]
        Number of repeating values along each axis.
    shape : ShapeLike
        Shape of the target tensor. If the shape does not match after stretching,
        remaining values are filled with zeroes.
    axes : tuple[int, int], optional
        Axes along which to stretch the tensor, by default (-2, -1).

    Returns
    -------
    Tensor
        Stretched out tensor.
    """
    st1, st2 = stretches
    ax1, ax2 = axes
    x_str = x.repeat(st1, ax1).repeat(st2, ax2)
    return x_str if x_str.shape == shape else x_str.pad_to_shape(shape)
