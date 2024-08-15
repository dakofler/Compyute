"""Tensor selection and filter operations."""

from typing import Optional

from ..base_tensor import ShapeError, Tensor, _AxisLike, tensor

__all__ = ["argmax", "get_diagonal", "topk", "tril", "triu", "unique"]


def argmax(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Returns the indices of maximum values along a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : _AxisLike, optional
        Axes, along which the maximum value is located. Defaults to ``None``.
        If ``None`` it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions. Defaults to ``False``.
        If false the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing indices.
    """
    return tensor(x.engine.argmax(x.data, axis=axis, keepdims=keepdims))


def get_diagonal(x: Tensor, d: int = 0) -> Tensor:
    """Extract a diagonal or construct a diagonal tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    d : int, optional
        | Index of the diagonal. Defaults to ``0``.
        | ``0``: main diagonal
        | ``> 0``: above main diagonal
        | ``< 0``: below main diagonal

    Returns
    -------
    Tensor
        The extracted diagonal or constructed diagonal tensor.
    """
    if x.n_axes < 2:
        raise ShapeError("Input tensor must have at least 2 dimensions.")
    return Tensor(x.engine.diag(x.data, k=d))


def topk(x: Tensor, k: int, axis: _AxisLike = -1) -> tuple[Tensor, Tensor]:
    """Returns the k largest elements along a given axis.
    Implementation by https://hippocampus-garden.com/numpy_topk/.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    k : int
        Number of top elements to select.
    axis : AxisLike
        Axis, along which the top elements are selected. Defaults to ``-1``.

    Returns
    -------
    tuple[Tensor, Tensor]
        Tuple containing the top k elements and their indices.
    """
    ind = x.engine.argpartition(-x.data, k, axis=axis)
    ind = x.engine.take(ind, x.engine.arange(k), axis=axis)
    data = x.engine.take_along_axis(-x.data, ind, axis=axis)

    # sort within k elements
    ind_part = x.engine.argsort(data, axis=axis)
    ind = x.engine.take_along_axis(ind, ind_part, axis=axis)

    val = x.engine.take_along_axis(-data, ind_part, axis=axis)
    return tensor(val), tensor(ind)


def tril(x: Tensor, d: int = 0) -> Tensor:
    """Returns the lower triangle of a tensor below the
    d-th diagonal of the last two dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    d : int, optional
        | Index of the diagonal. Defaults to ``0``.
        | ``0``: main diagonal
        | ``> 0``: above main diagonal
        | ``< 0``: below main diagonal

    Returns
    -------
    Tensor
        Lower triangle tensor.
    """
    if x.n_axes < 2:
        raise ShapeError("Input tensor must have at least 2 dimensions.")
    return Tensor(x.engine.tril(x.data, k=d))


def triu(x: Tensor, d: int = 0) -> Tensor:
    """Returns the upper triangle of a tensor above the
    d-th diagonal of the last two dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    d : int, optional
        | Index of the diagonal. Defaults to ``0``.
        | ``0``: main diagonal
        | ``> 0``: above main diagonal
        | ``< 0``: below main diagonal

    Returns
    -------
    Tensor
        Upper triangle tensor.
    """
    if x.n_axes < 2:
        raise ShapeError("Input tensor must have at least 2 dimensions.")
    return Tensor(x.engine.triu(x.data, k=d))


def unique(x: Tensor) -> Tensor:
    """Returns the unique ordered values of the tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing unique values.
    """
    return tensor(x.engine.unique(x.data))
