"""Tensor functions module"""

import operator
from functools import reduce
from typing import Iterable, Optional, Sequence

from .engine import get_engine
from .tensors import Tensor
from .types import AxisLike, DeviceLike, DtypeLike, ScalarLike, ShapeLike

__all__ = [
    "arange",
    "linspace",
    "empty",
    "zeros",
    "ones",
    "full",
    "empty_like",
    "zeros_like",
    "ones_like",
    "full_like",
    "identity",
    "diagonal",
    "maximum",
    "minimum",
    "concatenate",
    "stack",
    "tensorsum",
    "tensorprod",
]


def arange(
    stop: int,
    start: int = 0,
    step: int | float = 1,
    dtype: Optional[DtypeLike] = None,
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
    dtype: DtypeLike, optional
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
    dtype: Optional[DtypeLike] = None,
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
    dtype: DtypeLike, optional
        Datatype of the tensor data, by default None.
    device: DeviceLike, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor of evenly spaced samples.
    """
    return Tensor(get_engine(device).linspace(start, stop, num, dtype=dtype))


def empty(
    shape: ShapeLike, dtype: Optional[DtypeLike] = None, device: DeviceLike = "cpu"
) -> Tensor:
    """Returns an tensor with uninitialized values.

    Parameters
    ----------
    shape: ShapeLike
        Shape of the new tensor.
    dtype: DtypeLike, optional
        Datatype of the tensor data, by default None.
    device: DeviceLike, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with uninitialized values.
    """
    return Tensor(get_engine(device).empty(shape=shape, dtype=dtype))


def zeros(
    shape: ShapeLike, dtype: Optional[DtypeLike] = None, device: DeviceLike = "cpu"
) -> Tensor:
    """Returns a tensor of a given shape with all values being zero.

    Parameters
    ----------
    shape: ShapeLike
        Shape of the new tensor.
    dtype: DtypeLike, optional
        Datatype of the tensor data, by default None.
    device: DeviceLike, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with all values being zero.
    """
    return Tensor(get_engine(device).zeros(shape, dtype=dtype))


def ones(shape: ShapeLike, dtype: Optional[DtypeLike] = None, device: DeviceLike = "cpu") -> Tensor:
    """Returns a tensor of a given shape with all values being one.

    Parameters
    ----------
    shape: ShapeLike
        Shape of the new tensor.
    dtype: DtypeLike, optional
        Datatype of the tensor data, by default None.
    device: DeviceLike, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    return Tensor(get_engine(device).ones(shape, dtype=dtype))


def full(
    shape: ShapeLike,
    value: ScalarLike,
    dtype: Optional[DtypeLike] = None,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Returns a tensor of a given shape with all values being one.

    Parameters
    ----------
    shape: ShapeLike
        Shape of the new tensor.
    value: ScalarLike
        Value to fill the tensor.
    dtype: DtypeLike, optional
        Datatype of the tensor data, by default None.
    device: DeviceLike, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    return Tensor(get_engine(device).full(shape, value, dtype=dtype))


def empty_like(x: Tensor) -> Tensor:
    """Returns a tensor based on a given other tensor with uninitialized values.

    Parameters
    ----------
    x : Tensor
        Tensor whose shape, dtype and device is used.

    Returns
    -------
    Tensor
        Tensor with uninitialized values.
    """
    return empty(x.shape, dtype=x.dtype, device=x.device)


def zeros_like(x: Tensor) -> Tensor:
    """Returns a tensor based on a given other tensor with all values being zero.

    Parameters
    ----------
    x : Tensor
        Tensor whose shape, dtype and device is used.

    Returns
    -------
    Tensor
        Tensor with all values being zero.
    """
    return zeros(x.shape, dtype=x.dtype, device=x.device)


def ones_like(x: Tensor) -> Tensor:
    """Returns a tensor based on a given other tensor with all values being one.

    Parameters
    ----------
    x : Tensor
        Tensor whose shape, dtype and device is used.

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    return ones(x.shape, dtype=x.dtype, device=x.device)


def full_like(x: Tensor, value: ScalarLike) -> Tensor:
    """Returns a tensor of a given shape with all values being one.

    Parameters
    ----------
    x : Tensor
        Tensor whose shape, dtype and device is used.
    value: ScalarLike
        Value to fill the tensor.

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    return full(x.shape, value=value, dtype=x.dtype, device=x.device)


def identity(n: int, dtype: Optional[DtypeLike] = None, device: DeviceLike = "cpu") -> Tensor:
    """Returns a diagonal tensor of shape (n, n).

    Parameters
    ----------
    n: int
        Size of the new tensor. The shape will be (n, n).
    dtype: DtypeLike, optional
        Datatype of the tensor data, by default None.
    device: DeviceLike, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Diagonal tensor.
    """
    return Tensor(get_engine(device).identity(n, dtype=dtype))


def diagonal(x: Tensor) -> Tensor:
    """Expands a tensor by turning the last dim into a diagonal matrix.

    Parameters
    ----------
    x: Tensor
        Tensor to diagonalize.

    Returns
    -------
    Tensor
        The diagonal tensor.
    """
    return x.insert_dim(-1) * identity(x.shape[-1])


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


def concatenate(tensors: Sequence[Tensor], axis: AxisLike = -1) -> Tensor:
    """Returns a new tensor by joining a sequence of tensors along a given axis.

    Parameters
    ----------
    tensors : Sequence[Tensor]
        Sequence of Tensors to be joined.
    axis : AxisLike, optional
        Axis along which to join the tensors, by default -1.

    Returns
    -------
    Tensor
        Concatenated tensor.
    """
    device = tensors[0].device
    return Tensor(get_engine(device).concatenate([t.data for t in tensors], axis=axis))


def stack(tensors: Sequence[Tensor], axis: AxisLike = 0) -> Tensor:
    """Returns a new tensor by stacking a sequence of tensors along a given axis.

    Parameters
    ----------
    tensors : Sequence[Tensor]
        Sequence of Tensors to be stacked.
    axis : AxisLike, optional
        Axis along which to stack the tensors, by default 0.

    Returns
    -------
    Tensor
        Stacked tensor.
    """
    device = tensors[0].device
    return Tensor(get_engine(device).stack([t.data for t in tensors], axis=axis))


def tensorsum(tensors: Iterable[Tensor | ScalarLike]) -> Tensor | ScalarLike:
    """Sums the elements of an iterable element-wise over the first axis.

    Parameters
    ----------
    tensors : Iterable[Tensor | ScalarLike]
        Iterable of values to be summed.

    Returns
    -------
    Tensor | ScalarLike
        Tensor containing element-wise sums or scalar.
    """
    return reduce(operator.add, tensors)


def tensorprod(tensors: Iterable[Tensor | ScalarLike]) -> Tensor | ScalarLike:
    """Multiplies the elements of an iterable element-wise over the first axis.

    Parameters
    ----------
    tensors : Iterable[Tensor | ScalarLike]
        Iterable of values to be multiplied.

    Returns
    -------
    Tensor | ScalarLike
        Tensor containing element-wise products or scalar.
    """
    return reduce(operator.mul, tensors)
