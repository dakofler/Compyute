"""Tensor functions module"""

from typing import Optional
from .engine import get_engine
from .tensor import Tensor
from .types import AxisLike, DeviceLike, DtypeLike, ScalarLike, ShapeLike


__all__ = [
    "arange",
    "linspace",
    "empty",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    "empty_like",
    "eye",
    "maximum",
    "minimum",
    "prod",
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


def eye(n: int, dtype: Optional[DtypeLike] = None, device: DeviceLike = "cpu") -> Tensor:
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
    return Tensor(get_engine(device).eye(n, dtype=dtype))


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


def prod(x: list[int | float]) -> int | float:
    """Returns the product of a sequence of elements.

    Parameters
    ----------
    x : list[int | float]
        list of elements.

    Returns
    -------
    int | float
        Product of elements.
    """
    return get_engine("cpu").prod(x).item()


def concatenate(tensors: list[Tensor], axis: AxisLike = -1) -> Tensor:
    """Returns a new tensor by joining a sequence of tensors along a given axis.

    Parameters
    ----------
    tensors : list[Tensor]
        List of Tensors to be joined.
    axis : AxisLike, optional
        Axis along which to join the tensors, by default -1.

    Returns
    -------
    Tensor
        Concatenated tensor.
    """
    device = tensors[0].device
    return Tensor(get_engine(device).concatenate([t.data for t in tensors], axis=axis))


def stack(tensors: list[Tensor], axis: AxisLike = 0) -> Tensor:
    """Returns a new tensor by stacking a sequence of tensors along a given axis.

    Parameters
    ----------
    tensors : list[Tensor]
        List of Tensors to be stacked.
    axis : AxisLike, optional
        Axis along which to stack the tensors, by default 0.

    Returns
    -------
    Tensor
        Stacked tensor.
    """
    device = tensors[0].device
    return Tensor(get_engine(device).stack([t.data for t in tensors], axis=axis))


def tensorsum(tensors: list[Tensor], axis: AxisLike = 0) -> Tensor:
    """Sums a sequence of tensors element-wise.

    Parameters
    ----------
    tensors : list[Tensor]
        List of Tensors to be joined.
    axis : AxisLike, optional
        Axis along which to join the tensors, by default -1.

    Returns
    -------
    Tensor
        Tensor containing element-wise sums.
    """
    return stack(tensors=tensors, axis=axis).sum(axis=axis)


def tensorprod(tensors: list[Tensor], axis: AxisLike = 0) -> Tensor:
    """Multiplies a sequence of tensors element-wise.

    Parameters
    ----------
    tensors : list[Tensor]
        List of Tensors to be joined.
    axis : AxisLike, optional
        Axis along which to join the tensors, by default -1.

    Returns
    -------
    Tensor
        Tensor containing element-wise products.
    """
    return stack(tensors=tensors, axis=axis).prod(axis=axis)
