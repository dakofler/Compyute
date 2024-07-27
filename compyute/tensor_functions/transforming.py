"""Tensor transformation functions module"""

from typing import Optional

from ..base_tensor import Tensor, _AxisLike, _ShapeLike, tensor
from ..dtypes import _DtypeLike
from ..engine import get_engine

__all__ = [
    "sum",
    "prod",
    "mean",
    "var",
    "std",
    "min",
    "max",
    "round",
    "exp",
    "log",
    "log10",
    "log2",
    "sin",
    "sinh",
    "cos",
    "cosh",
    "tan",
    "tanh_",
    "sech",
    "abs",
    "sqrt",
    "fft1d",
    "ifft1d",
    "fft2d",
    "ifft2d",
    "real",
    "clip",
    "histogram",
]


def sum(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Computes the sum of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the sum is computed, by default None.
        If None it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions, by default False.
        If false the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the sum of elements.
    """
    return tensor(x.data.sum(axis=axis, keepdims=keepdims))


def prod(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Computes the product of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the product is computed, by default None.
        If None it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions, by default False.
        If false the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the product of elements.
    """
    return tensor(x.data.prod(axis=axis, keepdims=keepdims))


def mean(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Computes the mean of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the mean is computed, by default None.
        If None it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions, by default False.
        If false the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the mean of elements.
    """
    return tensor(x.data.mean(axis=axis, keepdims=keepdims))


def var(
    x: Tensor, axis: Optional[_AxisLike] = None, ddof: int = 0, keepdims: bool = False
) -> Tensor:
    """Computes the variance of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the variance is computed, by default None.
        If None it is computed over the flattened tensor.
    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof,
        where N represents the number of elements, by default 0.
    keepdims : bool, optional
        Whether to keep the tensors dimensions, by default False.
        If false the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the variance of elements.
    """
    return tensor(x.data.var(axis=axis, ddof=ddof, keepdims=keepdims))


def std(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Computes the standard deviation of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the standard deviation is computed, by default None.
        If None it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions, by default False.
        If false the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the standard deviation of elements.
    """
    return tensor(x.data.std(axis=axis, keepdims=keepdims))


def min(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Computes the minimum of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the minimum is computed, by default None.
        If None it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions, by default False.
        If false the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the minimum of elements.
    """
    return tensor(x.data.min(axis=axis, keepdims=keepdims))


def max(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Computes the maximum of tensor elements over a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        Axis over which the maximum is computed, by default None.
        If none it is computed over the flattened tensor.
    keepdims : bool, optional
        Whether to keep the tensors dimensions, by default False.
        If false the tensor is collapsed along the given axis.

    Returns
    -------
    Tensor
        Tensor containing the maximum of elements.
    """
    return tensor(x.data.max(axis=axis, keepdims=keepdims))


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
    return Tensor(x.data.round(decimals))


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
    return Tensor(get_engine(x.device).exp(x.data))


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
    return Tensor(get_engine(x.device).log(x.data))


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
    return Tensor(get_engine(x.device).log10(x.data))


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
    return Tensor(get_engine(x.device).log2(x.data))


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
    return Tensor(get_engine(x.device).sin(x.data))


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
    return Tensor(get_engine(x.device).sinh(x.data))


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
    return Tensor(get_engine(x.device).cos(x.data))


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
    return Tensor(get_engine(x.device).cosh(x.data))


def tan(x: Tensor) -> Tensor:
    """Computes the element-wise tangent of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    """
    return Tensor(get_engine(x.device).tan(x.data))


def tanh_(x: Tensor) -> Tensor:
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
    return Tensor(get_engine(x.device).tanh(x.data))


def sech(x: Tensor) -> Tensor:
    """Computes the element-wise secant of a tensor..

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
    return Tensor(get_engine(x.device).abs(x.data))


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
    return Tensor(get_engine(x.device).sqrt(x.data))


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
        Length of the transformed axis of the output, by default None.
    axis : int, optional
        Axis over which to compute the FFT, by default -1.
    dtype : _DtypeLike, optional
        Complex datatype of the output tensor, by default None.

    Returns
    -------
    Tensor
        Complex tensor containing the 1D FFT.
    """
    return tensor(
        get_engine(x.device).fft.fft(x.data, n=n, axis=axis), device=x.device, dtype=dtype
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
        Length of the transformed axis of the output, by default None.
    axis : int, optional
        Axis over which to compute the inverse FFT, by default -1.
    dtype : _DtypeLike, optional
        Complex datatype of the output tensor, by default None.

    Returns
    -------
    Tensor
        Float tensor containing the inverse 1D FFT.
    """
    return tensor(
        get_engine(x.device).fft.ifft(x.data, n=n, axis=axis), device=x.device, dtype=dtype
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
        Shape (length of each transformed axis) of the output, by default None.
    axes : tuple[int, int], optional
        Axes over which to compute the FFT, by default (-2, -1).
    dtype : _DtypeLike, optional
        Complex datatype of the output tensor, by default None.

    Returns
    -------
    Tensor
        Complex tensor containing the 2D FFT.
    """
    return tensor(
        get_engine(x.device).fft.fft2(x.data, s=s, axes=axes), device=x.device, dtype=dtype
    )


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
        Shape (length of each transformed axis) of the output, by default None.
    axes : tuple[int, int], optional
        Axes over which to compute the inverse FFT, by default (-2, -1).
    dtype : _DtypeLike, optional
        Complex datatype of the output tensor, by default None.

    Returns
    -------
    Tensor
        Float tensor containing the inverse 2D FFT.
    """
    return tensor(
        get_engine(x.device).fft.ifft2(x.data, s=s, axes=axes), device=x.device, dtype=dtype
    )


def real(x: Tensor, dtype: Optional[_DtypeLike] = None) -> Tensor:
    """Returns the real part of a complex tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dtype : DtypeLike, optional
        Datatype of the output tensor, by default None.

    Returns
    -------
    Tensor
        Tensor containing real values.
    """
    return tensor(get_engine(x.device).real(x.data), device=x.device, dtype=dtype)


def imag(x: Tensor, dtype: Optional[_DtypeLike] = None) -> Tensor:
    """Returns the imaginary part of a complex tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dtype : DtypeLike, optional
        Datatype of the output tensor, by default None.

    Returns
    -------
    Tensor
        Tensor containing imaginary values.
    """
    return tensor(get_engine(x.device).imag(x.data), device=x.device, dtype=dtype)


def clip(x: Tensor, min_value: Optional[float] = None, max_value: Optional[float] = None) -> Tensor:
    """Returns a tensor with values clipped to min_value and max_value.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    min_value : float, optional
        Lower bound, by default None. If None, no clipping is performed on this edge.
    max_value : float
        Upper bound, by default None. If None, no clipping is performed on this edge.

    Returns
    -------
    Tensor
        Tensor containing clipped values.
    """
    return Tensor(get_engine(x.device).clip(x.data, min_value, max_value))


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
        If int, defines the number of equal-width bins.
        If tensor, defines the sequence of bin edges including the rightmost edge.
        By default 10.
    range : tuple[float, float], optional
        Defines the range of the bins, by default None.
    density : bool, optional
        Whether to compute the density instead of the count, by default False.
    weights : Tensor, optional
        Each value in input contributes its associated weight
        towards its bin’s result, by default None.

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
    return Tensor(hist), Tensor(bin_edges)
