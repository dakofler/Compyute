"""Tensor computation functions module"""

import operator
from functools import reduce
from typing import Iterable, Iterator

from ..base_tensor import Tensor, tensor
from ..dtypes import _ScalarLike
from ..engine import get_engine

__all__ = ["maximum", "minimum", "tensorsum", "tensorprod", "inner", "outer", "einsum", "dot"]


def maximum(a: Tensor, b: Tensor | _ScalarLike) -> Tensor:
    """Returns a new tensor containing the element-wise maximum of two tensors/scalars.

    Parameters
    ----------
    a : Tensor
        Tensor.
    b : Tensor | _ScalarLike
        Tensor or scalar.

    Returns
    -------
    Tensor
        Tensor containing the element-wise maxima.
    """
    _b = b.data if isinstance(b, Tensor) else b
    return Tensor(get_engine(a.device).maximum(a.data, _b))


def minimum(a: Tensor, b: Tensor | _ScalarLike) -> Tensor:
    """Returns a new tensor containing the element-wise minimum of two tensors/scalars.

    Parameters
    ----------
    a : Tensor
        Tensor.
    b : Tensor | _ScalarLike
        Tensor or scalar.

    Returns
    -------
    Tensor
        Tensor containing the element-wise minima.
    """
    _b = b.data if isinstance(b, Tensor) else b
    return Tensor(get_engine(a.device).minimum(a.data, _b))


def tensorsum(tensors: Iterable[Tensor] | Iterator[Tensor]) -> Tensor:
    """Sums the elements of tensors element-wise over the first axis.

    Parameters
    ----------
    tensors : Iterable[Tensor] | Iterator[Tensor]
        Iterable or Iterator of tensors to be summed.

    Returns
    -------
    Tensor
        Tensor containing element-wise sums.
    """
    return reduce(operator.add, tensors)


def tensorprod(tensors: Iterable[Tensor] | Iterator[Tensor]) -> Tensor:
    """Multiplies the elements of tensors element-wise over the first axis.

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


def inner(*tensors: Tensor) -> Tensor:
    """Returns the inner product of tensors.

    Parameters
    ----------
    *tensors : Tensor
        Tensors to compute the inner product of.

    Returns
    -------
    Tensor
        Inner product.
    """
    device = tensors[0].device
    return tensor(get_engine(device).inner(*[t.data for t in tensors]))


def outer(*tensors: Tensor) -> Tensor:
    """Returns the outer product of tensors.

    Parameters
    ----------
    *args : Tensor
        Tensors to compute the outer product of.

    Returns
    -------
    Tensor
        Outer product.
    """
    device = tensors[0].device
    return Tensor(get_engine(device).outer(*[t.data for t in tensors]))


def dot(x: Tensor, y: Tensor) -> Tensor:
    """Dot product of two tensors.

    Parameters
    ----------
    x : Tensor
        First tensor.
    y : Tensor
        Second tensor.

    Returns
    -------
    Tensor
        Dot product of the tensors.
    """
    if x.ndim != 1 or y.ndim != 1:
        raise AttributeError("Inputs must be 1D-tensors.")
    return inner(x, y)


def einsum(subscripts, *tensors: Tensor) -> Tensor:
    """Evaluates the Einstein summation.

    Parameters
    ----------
    subscriptsstr : str
        Specifies the subscripts for summation as comma separated list of subscript labels.
        An implicit (classical Einstein summation) calculation is performed unless the explicit
        indicator ‘->’ is included as well as subscript labels of the precise output form.
    *tensors : Tensor
        Tensors to compute the outer product of.

    Returns
    -------
    Tensor
        Result based on the Einstein summation.
    """
    device = tensors[0].device
    return tensor(get_engine(device).einsum(subscripts, *[t.data for t in tensors]))
