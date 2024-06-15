"""Tensor computation functions module"""

import operator
from functools import reduce
from typing import Iterable

from .._tensor import Tensor, _as_tensor
from .._types import _ScalarLike
from ..engine import _get_engine

__all__ = ["maximum", "minimum", "tensorsum", "tensorprod", "inner", "outer", "einsum", "dot"]


def maximum(a: Tensor | _ScalarLike, b: Tensor | _ScalarLike) -> Tensor:
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

    return Tensor(_get_engine(device).maximum(_a, _b))


def minimum(a: Tensor | _ScalarLike, b: Tensor | _ScalarLike) -> Tensor:
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

    return Tensor(_get_engine(device).minimum(_a, _b))


def tensorsum(tensors: Iterable[Tensor | _ScalarLike]) -> Tensor | _ScalarLike:
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


def tensorprod(tensors: Iterable[Tensor | _ScalarLike]) -> Tensor | _ScalarLike:
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
    return _as_tensor(_get_engine(device).inner(*[t.data for t in tensors]))


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
    return Tensor(_get_engine(device).outer(*[t.data for t in tensors]))


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
    return _as_tensor(_get_engine(device).einsum(subscripts, *[t.data for t in tensors]))
