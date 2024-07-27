"""Tensor combination functions module"""

from typing import Sequence

from ..base_tensor import Tensor, _AxisLike
from ..engine import get_engine

__all__ = [
    "append",
    "concatenate",
    "split",
    "stack",
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
