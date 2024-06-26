"""Tensor selection functions module"""

from typing import Optional

from ..base_tensor import Tensor, _as_tensor
from ..engine import _get_engine
from ..types import _AxisLike

__all__ = ["argmax", "get_diagonal", "tril", "triu", "unique"]


def argmax(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Returns the indices of maximum values along a given axis.

    Parameters
    ----------
    x: Tensor
        Input tensor.
    axis : Optional[AxisLike] = None
        Axes, along which the maximum value is located, by default None.
        If None it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions, by default False.
        If false the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing indices.
    """
    return _as_tensor(_get_engine(x.device).argmax(x.data, axis=axis, keepdims=keepdims))


def get_diagonal(x: Tensor, d: int = 0) -> Tensor:
    """Extract a diagonal or construct a diagonal tensor.

    Parameters
    ----------
    x: Tensor
        Input tensor.
    d : int, optional
        Index of the diagonal, by default 0.
        `0`: main diagonal
        `> 0`: above main diagonal
        `< 0`: below main diagonal

    Returns
    -------
    Tensor
        The extracted diagonal or constructed diagonal tensor.
    """
    return Tensor(_get_engine(x.device).diag(x.data, k=d))


def tril(x: Tensor, d: int = 0) -> Tensor:
    """Returns the lower triangle of a tensor below the
    d-th diagonal of the last two dimensions.

    Parameters
    ----------
    x: Tensor
        Input tensor.
    d : int, optional
        Index of the diagonal, by default 0.
        `0`: main diagonal
        `> 0`: above main diagonal
        `< 0`: below main diagonal

    Returns
    -------
    Tensor
        Lower triangle tensor.
    """
    return Tensor(_get_engine(x.device).tril(x.data, k=d))


def triu(x: Tensor, d: int = 0) -> Tensor:
    """Returns the upper triangle of a tensor above the
    d-th diagonal of the last two dimensions.

    Parameters
    ----------
    x: Tensor
        Input tensor.
    d : int, optional
        Index of the diagonal, by default 0.
        `0`: main diagonal
        `> 0`: above main diagonal
        `< 0`: below main diagonal

    Returns
    -------
    Tensor
        Upper triangle tensor.
    """
    return Tensor(_get_engine(x.device).triu(x.data, k=d))


def unique(x: Tensor) -> Tensor:
    """Returns the unique ordered values of the tensor."""
    return Tensor(_get_engine(x.device).unique(x.data))
