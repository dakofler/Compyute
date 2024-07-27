"""Tensor creation and combination functions."""

from typing import Optional, Sequence

from ..base_tensor import Tensor, _AxisLike, _ShapeLike
from ..dtypes import _DtypeLike, _ScalarLike, dtype_to_str
from ..engine import Device, _DeviceLike, get_engine

__all__ = [
    "append",
    "arange",
    "concatenate",
    "empty",
    "empty_like",
    "full",
    "full_like",
    "identity",
    "linspace",
    "ones",
    "ones_like",
    "split",
    "stack",
    "zeros",
    "zeros_like",
]


def append(x: Tensor, values: Tensor, axis: int = -1) -> Tensor:
    """Returns a copy of the tensor with appended values.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    values : Tensor
        Values to append.
    axis : int, optional
        Axis alowng which to append the values, by default -1.

    Returns
    -------
    Tensor
        Tensor containing appended values.
    """
    return Tensor(get_engine(x.device).append(x.data, values.data, axis=axis))


def arange(
    stop: int,
    start: int = 0,
    step: int | float = 1,
    dtype: Optional[_DtypeLike] = None,
    device: _DeviceLike = Device.CPU,
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
    dtype : DtypeLike, optional
        Datatype of the tensor data, by default None.
    device : DeviceLike, optional
        The device the tensor is stored on, by default Device.CPU.

    Returns
    -------
    Tensor
        Tensor of evenly spaced samples.
    """
    dtype = dtype_to_str(dtype) if dtype is not None else dtype
    return Tensor(get_engine(device).arange(start, stop, step, dtype=dtype))


def concatenate(tensors: Sequence[Tensor], axis: _AxisLike = -1) -> Tensor:
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


def empty(
    shape: _ShapeLike, dtype: Optional[_DtypeLike] = None, device: _DeviceLike = Device.CPU
) -> Tensor:
    """Returns an tensor with uninitialized values.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the new tensor.
    dtype : DtypeLike, optional
        Datatype of the tensor data, by default None.
    device : DeviceLike, optional
        The device the tensor is stored on, by default Device.CPU.

    Returns
    -------
    Tensor
        Tensor with uninitialized values.
    """
    dtype = dtype_to_str(dtype) if dtype is not None else dtype
    return Tensor(get_engine(device).empty(shape=shape, dtype=dtype))


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


def full(
    shape: _ShapeLike,
    value: _ScalarLike,
    dtype: Optional[_DtypeLike] = None,
    device: _DeviceLike = Device.CPU,
) -> Tensor:
    """Returns a tensor of a given shape with all values being one.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the new tensor.
    value : ScalarLike
        Value to fill the tensor.
    dtype : DtypeLike, optional
        Datatype of the tensor data, by default None.
    device : DeviceLike, optional
        The device the tensor is stored on, by default Device.CPU.

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    dtype = dtype_to_str(dtype) if dtype is not None else dtype
    return Tensor(get_engine(device).full(shape, value, dtype=dtype))


def full_like(x: Tensor, value: _ScalarLike) -> Tensor:
    """Returns a tensor of a given shape with all values being one.

    Parameters
    ----------
    x : Tensor
        Tensor whose shape, dtype and device is used.
    value : ScalarLike
        Value to fill the tensor.

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    return full(x.shape, value=value, dtype=x.dtype, device=x.device)


def identity(
    n: int, dtype: Optional[_DtypeLike] = None, device: _DeviceLike = Device.CPU
) -> Tensor:
    """Returns a diagonal tensor of shape (n, n).

    Parameters
    ----------
    n : int
        Size of the new tensor. The shape will be (n, n).
    dtype : DtypeLike, optional
        Datatype of the tensor data, by default None.
    device : DeviceLike, optional
        The device the tensor is stored on, by default Device.CPU.

    Returns
    -------
    Tensor
        Diagonal tensor.
    """
    dtype = dtype_to_str(dtype) if dtype is not None else dtype
    return Tensor(get_engine(device).identity(n, dtype=dtype))


def linspace(
    start: float,
    stop: float,
    num: int,
    dtype: Optional[_DtypeLike] = None,
    device: _DeviceLike = Device.CPU,
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
    dtype : DtypeLike, optional
        Datatype of the tensor data, by default None.
    device : DeviceLike, optional
        The device the tensor is stored on, by default Device.CPU.

    Returns
    -------
    Tensor
        Tensor of evenly spaced samples.
    """
    dtype = dtype_to_str(dtype) if dtype is not None else dtype
    return Tensor(get_engine(device).linspace(start, stop, num, dtype=dtype))


def ones(
    shape: _ShapeLike, dtype: Optional[_DtypeLike] = None, device: _DeviceLike = Device.CPU
) -> Tensor:
    """Returns a tensor of a given shape with all values being one.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the new tensor.
    dtype : DtypeLike, optional
        Datatype of the tensor data, by default None.
    device : DeviceLike, optional
        The device the tensor is stored on, by default Device.CPU.

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    dtype = dtype_to_str(dtype) if dtype is not None else dtype
    return Tensor(get_engine(device).ones(shape, dtype=dtype))


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


def split(x: Tensor, splits: int | Sequence[int], axis: int = -1) -> list[Tensor]:
    """Returns a list of new tensors by splitting the tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    splits : int | list[int]
        `int`: tensor is split into n equally sized tensors.
        `Sequence[int]`: tensor is split at the given indices.
    axis : int, optional
        Axis along which to split the tensor, by default -1.

    Returns
    -------
    list[Tensor]
        List of tensors containing the split data.
    """
    return [Tensor(s) for s in get_engine(x.device).split(x.data, splits, axis=axis)]


def stack(tensors: Sequence[Tensor], axis: _AxisLike = 0) -> Tensor:
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


def zeros(
    shape: _ShapeLike, dtype: Optional[_DtypeLike] = None, device: _DeviceLike = Device.CPU
) -> Tensor:
    """Returns a tensor of a given shape with all values being zero.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the new tensor.
    dtype : DtypeLike, optional
        Datatype of the tensor data, by default None.
    device : DeviceLike, optional
        The device the tensor is stored on, by default Device.CPU.

    Returns
    -------
    Tensor
        Tensor with all values being zero.
    """
    dtype = dtype_to_str(dtype) if dtype is not None else dtype
    return Tensor(get_engine(device).zeros(shape, dtype=dtype))


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
