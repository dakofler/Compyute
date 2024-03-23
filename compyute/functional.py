"""Tensor functions module"""

import numpy
from .engine import get_engine
from .tensor import Tensor
from .types import DeviceLike, DtypeLike, ScalarLike, ShapeLike


__all__ = [
    "arange",
    "linspace",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    "empty",
    "maximum",
    "concatenate",
    "prod",
    "eye",
]


def arange(
    stop: int,
    start: int = 0,
    step: int | float = 1,
    dtype: DtypeLike | None = None,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Returns a tensor of evenly spaced values using a step size within
    a given interval [start, stop).

    Parameters
    ----------
    stop : float
        Stop value (not included).
    start : float, optional
        Start value, by default 0.
    step : int | float, optional
        Spacing between values, by default 1.
    dtype: DtypeLike | None, optional
        Datatype of the tensor data, by default None.
    device: DeviceLike, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor of evenly spaced samples.
    """
    return Tensor(get_engine(device).arange(start, stop, step, dtype=dtype))


def linspace(
    start: float,
    stop: float,
    num: int,
    dtype: DtypeLike | None = None,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Returns a tensor of num evenly spaced values within
    a given interval [start, stop].

    Parameters
    ----------
    start : float
        Start value.
    stop : float
        Stop value.
    num : int
        Number of samples.
    dtype: DtypeLike | None, optional
        Datatype of the tensor data, by default None.
    device: DeviceLike, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor of evenly spaced samples.
    """
    return Tensor(get_engine(device).linspace(start, stop, num, dtype=dtype))


def zeros(
    shape: ShapeLike, dtype: DtypeLike | None = None, device: DeviceLike = "cpu"
) -> Tensor:
    """Returns a tensor of a given shape with all values being zero.

    Parameters
    ----------
    ShapeLike
        Shape of the new tensor.
    dtype: DtypeLike | None, optional
        Datatype of the tensor data, by default None.
    device: DeviceLike, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with all values being zero.
    """
    return Tensor(get_engine(device).zeros(shape, dtype=dtype))


def ones(
    shape: ShapeLike, dtype: DtypeLike | None = None, device: DeviceLike = "cpu"
) -> Tensor:
    """Returns a tensor of a given shape with all values being one.

    Parameters
    ----------
    ShapeLike
        Shape of the new tensor.
    dtype: DtypeLike | None, optional
        Datatype of the tensor data, by default None.
    device: DeviceLike, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    return Tensor(get_engine(device).ones(shape, dtype=dtype))


def zeros_like(
    x: Tensor, dtype: DtypeLike | None = None, device: DeviceLike = "cpu"
) -> Tensor:
    """Returns a tensor based on the shape of a given other tensor with all values being zero.

    Parameters
    ----------
    x : Tensor
        Tensor whose shape is used.
    dtype: DtypeLike | None, optional
        Datatype of the tensor data, by default None.
    device: DeviceLike, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with all values being zero.
    """
    return zeros(x.shape, dtype=dtype, device=device)


def ones_like(
    x: Tensor, dtype: DtypeLike | None = None, device: DeviceLike = "cpu"
) -> Tensor:
    """Returns a tensor based on the shape of a given other tensor with all values being one.

    Parameters
    ----------
    x : Tensor
        Tensor whose shape is used.
    dtype: DtypeLike | None, optional
        Datatype of the tensor data, by default None.
    device: DeviceLike, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    return ones(x.shape, dtype=dtype, device=device)


def empty(dtype: DtypeLike | None = None, device: DeviceLike = "cpu") -> Tensor:
    """Returns an empty tensor.

    Parameters
    ----------
    dtype: DtypeLike | None, optional
        Datatype of the tensor data, by default None.
    device: DeviceLike, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Empty tensor.
    """
    return Tensor(get_engine(device).empty(0, dtype=dtype))


def maximum(a: Tensor | ScalarLike, b: Tensor | ScalarLike) -> Tensor:
    """Returns a new tensor containing the element-wise maximum of two tensors/scalars.

    Parameters
    ----------
    a : Tensor | ScalarLike
        Tensor or scalar.
    b : Tensor | ScalarLike
        Tensor or scalar.

    Returns
    -------
    Tensor
        Tensor containing the element-wise maxima.
    """
    device = "cpu"  # set cpu in case of two scalars

    if isinstance(a, Tensor):
        device = a.device
        _a = a.data
    else:
        _a = a

    if isinstance(b, Tensor):
        device = b.device
        _b = b.data
    else:
        _b = b

    return Tensor(get_engine(device).maximum(_a, _b))


def minimum(a: Tensor | ScalarLike, b: Tensor | ScalarLike) -> Tensor:
    """Returns a new tensor containing the element-wise minimum of two tensors/scalars.

    Parameters
    ----------
    a : Tensor | ScalarLike
        Tensor or scalar.
    b : Tensor | ScalarLike
        Tensor or scalar.

    Returns
    -------
    Tensor
        Tensor containing the element-wise minima.
    """
    device = "cpu"

    if isinstance(a, Tensor):
        device = a.device
        _a = a.data
    else:
        _a = a

    if isinstance(b, Tensor):
        device = b.device
        _b = b.data
    else:
        _b = b

    return Tensor(get_engine(device).minimum(_a, _b))


def concatenate(tensors: list[Tensor], axis: int = -1) -> Tensor:
    """Returns a new tensor by joins a sequence of tensors along a given axis.

    Parameters
    ----------
    tensors : list[Tensor]
        List of Tensors to be joined.
    axis : int, optional
        Axis along which to join the tensors, by default -1.

    Returns
    -------
    Tensor
        Concatenated tensor.
    """

    device = tensors[0].device
    return Tensor(get_engine(device).concatenate([t.data for t in tensors], axis=axis))


def prod(x: tuple[int, ...]) -> int:
    """Returns the product of tuple elements.

    Parameters
    ----------
    x : tuple[int, ...]
        Tuple of integers.

    Returns
    -------
    int
        Product of tuple elements.
    """
    return numpy.prod(x).item()


def eye(n: int, dtype: DtypeLike | None = None, device: DeviceLike = "cpu") -> Tensor:
    """Returns a diagonal tensor of shape (n, n).

    Parameters
    ----------
    n: int
        Size of the new tensor. The shape will be (n, n).
    dtype: DtypeLike | None, optional
        Datatype of the tensor data, by default None.
    device: DeviceLike, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Diagonal tensor.
    """
    return Tensor(get_engine(device).eye(n, dtype=dtype))
