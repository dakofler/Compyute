"""Tensor computation and transformation operations."""

import operator
from functools import reduce
from typing import Iterable, Iterator, Optional

from ..base_tensor import ShapeError, Tensor, _AxisLike, _ShapeLike, tensor
from ..dtypes import _DtypeLike, _ScalarLike
from ..engine import get_engine

__all__ = [
    "abs",
    "clip",
    "cos",
    "cosh",
    "dot",
    "einsum",
    "exp",
    "fft1d",
    "fft2d",
    "histogram",
    "inner",
    "ifft1d",
    "ifft2d",
    "log",
    "log2",
    "log10",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "norm",
    "outer",
    "prod",
    "real",
    "round",
    "sech",
    "sin",
    "sinh",
    "sqrt",
    "std",
    "tan",
    "tanh",
    "tensorprod",
    "tensorsum",
    "sum",
    "var",
]


def abs(x: Tensor) -> Tensor:
    """Computes the element-wise absolute value of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise absolute value.
    """
    return tensor(get_engine(x.device).abs(x.data))


def clip(x: Tensor, min_value: Optional[float] = None, max_value: Optional[float] = None) -> Tensor:
    """Returns a tensor with values clipped to min_value and max_value.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    min_value : float, optional
        Lower bound. Defaults to ``None``. If ``None``, no clipping is performed on this edge.
    max_value : float
        Upper bound. Defaults to ``None``. If ``None``, no clipping is performed on this edge.

    Returns
    -------
    Tensor
        Tensor containing clipped values.
    """
    return tensor(get_engine(x.device).clip(x.data, min_value, max_value))


def cos(x: Tensor) -> Tensor:
    """Computes the element-wise cosine of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise cosine.
    """
    return tensor(get_engine(x.device).cos(x.data))


def cosh(x: Tensor) -> Tensor:
    """Computes the element-wise hyperbolic cosine of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise hyperbolic cosine.
    """
    return tensor(get_engine(x.device).cosh(x.data))


def dot(x: Tensor, y: Tensor) -> Tensor:
    """Computes the dot product of two tensors.

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
        raise ShapeError("Inputs must be 1D-tensors.")
    return inner(x, y)


def einsum(subscripts: str, *tensors: Tensor) -> Tensor:
    """Computes the Einstein summation.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation as a comma separated list of subscript labels.
        An implicit (classical Einstein summation) calculation is performed unless the explicit
        indicator "->" is included as well as subscript labels of the precise output form.
    tensors : Tensor
        Tensors to evaluate the Einstein summation for.

    Returns
    -------
    Tensor
        Result of the Einstein summation.
    """
    device = tensors[0].device
    return tensor(get_engine(device).einsum(subscripts, *[t.data for t in tensors]))


def exp(x: Tensor) -> Tensor:
    """Computes the element-wise exponential of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise exponential.
    """
    return tensor(get_engine(x.device).exp(x.data))


def fft1d(
    x: Tensor,
    n: Optional[int] = None,
    axis: int = -1,
    dtype: Optional[_DtypeLike] = None,
) -> Tensor:
    """Computes the 1D Fast Fourier Transform over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    n : int, optional
        Length of the transformed axis of the output. Defaults to ``None``.
    axis : int, optional
        Axis over which to compute the FFT. Defaults to ``-1``.
    dtype : _DtypeLike, optional
        Complex datatype of the output tensor. Defaults to ``None``.

    Returns
    -------
    Tensor
        Complex tensor containing the 1D FFT.
    """
    return tensor(
        get_engine(x.device).fft.fft(x.data, n=n, axis=axis), device=x.device, dtype=dtype
    )


def fft2d(
    x: Tensor,
    s: Optional[_ShapeLike] = None,
    axes: tuple[int, int] = (-2, -1),
    dtype: Optional[_DtypeLike] = None,
) -> Tensor:
    """Computes the 2D Fast Fourier Transform over given axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    n : ShapeLike, optional
        Shape (length of each transformed axis) of the output. Defaults to ``None``.
    axes : tuple[int, int], optional
        Axes over which to compute the FFT. Defaults to ``(-2, -1)``.
    dtype : _DtypeLike, optional
        Complex datatype of the output tensor. Defaults to ``None``.

    Returns
    -------
    Tensor
        Complex tensor containing the 2D FFT.
    """
    return tensor(
        get_engine(x.device).fft.fft2(x.data, s=s, axes=axes), device=x.device, dtype=dtype
    )


def ifft1d(
    x: Tensor,
    n: Optional[int] = None,
    axis: int = -1,
    dtype: Optional[_DtypeLike] = None,
) -> Tensor:
    """Computes the inverse 1D Fast Fourier Transform over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    n : int, optional
        Length of the transformed axis of the output. Defaults to ``None``.
    axis : int, optional
        Axis over which to compute the inverse FFT. Defaults to ``-1``.
    dtype : _DtypeLike, optional
        Complex datatype of the output tensor. Defaults to ``None``.

    Returns
    -------
    Tensor
        Float tensor containing the inverse 1D FFT.
    """
    return tensor(
        get_engine(x.device).fft.ifft(x.data, n=n, axis=axis), device=x.device, dtype=dtype
    )


def imag(x: Tensor, dtype: Optional[_DtypeLike] = None) -> Tensor:
    """Returns the imaginary part of a complex tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dtype : DtypeLike, optional
        Datatype of the output tensor. Defaults to ``None``.

    Returns
    -------
    Tensor
        Tensor containing imaginary values.
    """
    return tensor(get_engine(x.device).imag(x.data), device=x.device, dtype=dtype)


def inner(*tensors: Tensor) -> Tensor:
    """Computes the inner product of two or more tensors.

    Parameters
    ----------
    *tensors : Tensor
        Tensors to compute the inner product of.

    Returns
    -------
    Tensor
        Tensor containing the inner product.
    """
    device = tensors[0].device
    return tensor(get_engine(device).inner(*[t.data for t in tensors]))


def ifft2d(
    x: Tensor,
    s: Optional[_ShapeLike] = None,
    axes: tuple[int, int] = (-2, -1),
    dtype: Optional[_DtypeLike] = None,
) -> Tensor:
    """Computes the inverse 2D Fast Fourier Transform over given axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    n : ShapeLike, optional
        Shape (length of each transformed axis) of the output. Defaults to ``None``.
    axes : tuple[int, int], optional
        Axes over which to compute the inverse FFT. Defaults to ``(-2, -1)``.
    dtype : _DtypeLike, optional
        Complex datatype of the output tensor. Defaults to ``None``.

    Returns
    -------
    Tensor
        Float tensor containing the inverse 2D FFT.
    """
    return tensor(
        get_engine(x.device).fft.ifft2(x.data, s=s, axes=axes), device=x.device, dtype=dtype
    )


def histogram(
    x: Tensor,
    bins: int | Tensor = 10,
    binrange: Optional[tuple[float, float]] = None,
    density: bool = False,
    weights: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Computes the histogram of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    bins : int | 1D Tensor
        The number of bins to use for the histogram. Defaults to 10.
            - ``int``:defines the number of equal-width bins.
            - ``Tensor``: defines the sequence of bin edges including the rightmost edge.

    range : tuple[float, float], optional
        Defines the range of the bins. Defaults to ``None``.
    density : bool, optional
        Whether to compute the density instead of the count. Defaults to ``False``.
    weights : Tensor, optional
        Each value in input contributes its associated weight
        towards its bin’s result. Defaults to ``None``.

    Returns
    -------
    Tensor
        Tensor containing the histogram values.
    Tensor
        Tensor containing the bin edges.
    """
    b = bins.data if isinstance(bins, Tensor) else bins
    w = weights.data if weights is not None else None
    hist, bin_edges = get_engine(x.device).histogram(
        x.data, bins=b, range=binrange, density=density, weights=w
    )
    return tensor(hist), tensor(bin_edges)


def log(x: Tensor) -> Tensor:
    """Computes the element-wise natural log of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise natural log.
    """
    return tensor(get_engine(x.device).log(x.data))


def log2(x: Tensor) -> Tensor:
    """Computes the element-wise log base 2 of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise log base 2.
    """
    return tensor(get_engine(x.device).log2(x.data))


def log10(x: Tensor) -> Tensor:
    """Computes the element-wise log base 10 of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise log base 10.
    """
    return tensor(get_engine(x.device).log10(x.data))


def max(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Computes the maximum of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the maximum is computed. Defaults to ``None``.
        If none it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions. Defaults to ``False``.
        if ``False`` the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the maximum of elements.
    """
    return tensor(x.data.max(axis=axis, keepdims=keepdims))


def maximum(x: Tensor, y: Tensor | _ScalarLike) -> Tensor:
    """Computes the element-wise maximum of two tensors or a tensor and a scalar.

    Parameters
    ----------
    x : Tensor
        First tensor.
    y : Tensor | _ScalarLike
        Second tensor or scalar.

    Returns
    -------
    Tensor
        Tensor containing the element-wise maximum.
    """
    _y = y.data if isinstance(y, Tensor) else y
    return tensor(get_engine(x.device).maximum(x.data, _y))


def mean(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Computes the mean of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the mean is computed. Defaults to ``None``.
        If ``None`` it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions. Defaults to ``False``.
        if ``False`` the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the mean of elements.
    """
    return tensor(x.data.mean(axis=axis, keepdims=keepdims))


def min(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Computes the minimum of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the minimum is computed. Defaults to ``None``.
        If ``None`` it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions. Defaults to ``False``.
        if ``False`` the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the minimum of elements.
    """
    return tensor(x.data.min(axis=axis, keepdims=keepdims))


def minimum(x: Tensor, y: Tensor | _ScalarLike) -> Tensor:
    """Computes the element-wise minimum of two tensors or a tensor and a scalar.

    Parameters
    ----------
    x : Tensor
        First tensor.
    y : Tensor | _ScalarLike
        Second tensor or scalar.

    Returns
    -------
    Tensor
        Tensor containing the element-wise minimum.
    """
    _y = y.data if isinstance(y, Tensor) else y
    return tensor(get_engine(x.device).minimum(x.data, _y))


def norm(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Computes the norm of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the norm is computed. Defaults to ``None``.
        If ``None`` it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions. Defaults to ``False``.
        if ``False`` the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the norm of elements.
    """
    return tensor(get_engine(x.device).linalg.norm(x.data, axis=axis, keepdims=keepdims))


def outer(*tensors: Tensor) -> Tensor:
    """Computes the outer product of two or more tensors.

    Parameters
    ----------
    *tensors : Tensor
        Tensors to compute the outer product of.

    Returns
    -------
    Tensor
        Tensor containing the outer product.
    """
    device = tensors[0].device
    return tensor(get_engine(device).outer(*[t.data for t in tensors]))


def prod(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Computes the product of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the product is computed. Defaults to ``None``.
        If ``None`` it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions. Defaults to ``False``.
        if ``False`` the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the product of elements.
    """
    return tensor(x.data.prod(axis=axis, keepdims=keepdims))


def real(x: Tensor, dtype: Optional[_DtypeLike] = None) -> Tensor:
    """Returns the real part of a complex tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dtype : DtypeLike, optional
        Datatype of the output tensor. Defaults to ``None``.

    Returns
    -------
    Tensor
        Tensor containing real values.
    """
    return tensor(get_engine(x.device).real(x.data), device=x.device, dtype=dtype)


def round(x: Tensor, decimals: int) -> Tensor:
    """Rounds tensor elements.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    decimals : int
        Decimal places of rounded values.

    Returns
    -------
    Tensor
        Tensor containing rounded elements.
    """
    return tensor(x.data.round(decimals))


def sech(x: Tensor) -> Tensor:
    """Computes the element-wise secant of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise secant.
    """
    return cosh(x) ** -1


def sin(x: Tensor) -> Tensor:
    """Computes the element-wise sine of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise sine.
    """
    return tensor(get_engine(x.device).sin(x.data))


def sinh(x: Tensor) -> Tensor:
    """Computes the element-wise hyperbolic sine of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise hyperbolic sine.
    """
    return tensor(get_engine(x.device).sinh(x.data))


def sqrt(x: Tensor) -> Tensor:
    """Computes the element-wise square root of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise square root.
    """
    return tensor(get_engine(x.device).sqrt(x.data))


def sum(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Computes the sum of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the sum is computed. Defaults to ``None``.
        If ``None`` it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions. Defaults to ``False``.
        if ``False`` the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the sum of elements.
    """
    return tensor(x.data.sum(axis=axis, keepdims=keepdims))


def std(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Computes the standard deviation of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the standard deviation is computed. Defaults to ``None``.
        If ``None`` it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions. Defaults to ``False``.
        if ``False`` the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the standard deviation of elements.
    """
    return tensor(x.data.std(axis=axis, keepdims=keepdims))


def tan(x: Tensor) -> Tensor:
    """Computes the element-wise tangent of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise tangent.
    """
    return tensor(get_engine(x.device).tan(x.data))


def tanh(x: Tensor) -> Tensor:
    """Computes the element-wise hyperbolic tangent of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise hyperbolic tangent.
    """
    return tensor(get_engine(x.device).tanh(x.data))


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
    x: Tensor, axis: Optional[_AxisLike] = None, ddof: int = 0, keepdims: bool = False
) -> Tensor:
    """Computes the variance of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the variance is computed. Defaults to ``None``.
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
    return tensor(x.data.var(axis=axis, ddof=ddof, keepdims=keepdims))
