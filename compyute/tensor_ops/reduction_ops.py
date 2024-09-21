"""Tensor reduction operations."""

import operator
from collections.abc import Iterable, Iterator
from functools import reduce
from typing import Optional

from ..tensors import AxisLike, Tensor

__all__ = [
    "all",
    "mean",
    "norm",
    "prod",
    "std",
    "tensorprod",
    "tensorsum",
    "sum",
    "var",
]


def all(
    x: Tensor, axis: Optional[AxisLike] = None, *, keepdims: bool = False
) -> Tensor:
    """Returns ``True`` if all elements in the tensor are ``True``.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the operation is computed. Defaults to ``None``.
        If ``None`` it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions. Defaults to ``False``.
        if ``False`` the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        ``True`` if all elements are ``True``.
    """
    return x.all(axis, keepdims=keepdims)


def mean(
    x: Tensor, axis: Optional[AxisLike] = None, *, keepdims: bool = False
) -> Tensor:
    """Computes the mean of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the operation is computed. Defaults to ``None``.
        If ``None`` it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions. Defaults to ``False``.
        if ``False`` the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the mean of elements.
    """
    return x.mean(axis, keepdims=keepdims)


def norm(
    x: Tensor, axis: Optional[AxisLike] = None, *, keepdims: bool = False
) -> Tensor:
    """Computes the norm of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the operation is computed. Defaults to ``None``.
        If ``None`` it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions. Defaults to ``False``.
        if ``False`` the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the norm of elements.
    """
    return Tensor(x.device.module.linalg.norm(x.data, axis=axis, keepdims=keepdims))


def prod(
    x: Tensor, axis: Optional[AxisLike] = None, *, keepdims: bool = False
) -> Tensor:
    """Computes the product of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the operation is computed. Defaults to ``None``.
        If ``None`` it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions. Defaults to ``False``.
        if ``False`` the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the product of elements.
    """
    return Tensor(x.data.prod(axis, keepdims=keepdims))


def std(
    x: Tensor, axis: Optional[AxisLike] = None, *, keepdims: bool = False
) -> Tensor:
    """Computes the standard deviation of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the operation is computed. Defaults to ``None``.
        If ``None`` it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions. Defaults to ``False``.
        if ``False`` the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the standard deviation of elements.
    """
    return x.std(axis, keepdims=keepdims)


def sum(
    x: Tensor, axis: Optional[AxisLike] = None, *, keepdims: bool = False
) -> Tensor:
    """Computes the sum of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the operation is computed. Defaults to ``None``.
        If ``None`` it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions. Defaults to ``False``.
        if ``False`` the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the sum of elements.
    """
    return x.sum(axis, keepdims=keepdims)


def tensorprod(tensors: Iterable[Tensor] | Iterator[Tensor]) -> Tensor:
    """Computes the element-wise product of any number of two or more tensors over their first axis.

    Parameters
    ----------
    tensors : Iterable[Tensor] | Iterator[Tensor]
        Iterable or Iterator of tensors to be multiplied.

    Returns
    -------
    Tensor
        Tensor containing element-wise products.
    """
    return reduce(operator.mul, tensors)


def tensorsum(tensors: Iterable[Tensor] | Iterator[Tensor]) -> Tensor:
    """Computes the element-wise sum of any number of two or more tensors over their first axis.

    Parameters
    ----------
    tensors : Iterable[Tensor] | Iterator[Tensor]
        Iterable or Iterator of tensors to be sumed.

    Returns
    -------
    Tensor
        Tensor containing element-wise sums.
    """
    return reduce(operator.add, tensors)


def var(
    x: Tensor, axis: Optional[AxisLike] = None, *, ddof: int = 0, keepdims: bool = False
) -> Tensor:
    """Computes the variance of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the operation is computed. Defaults to ``None``.
        If ``None`` it is computed over the flattened tensor.
    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is :math:`N - ddof`,
        where :math:`N` represents the number of elements. Defaults to ``0``.
    keepdims : bool, optional
        Whether to keep the tensors dimensions. Defaults to ``False``.
        If ``False`` the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the variance of elements.
    """
    return x.var(axis, ddof=ddof, keepdims=keepdims)
