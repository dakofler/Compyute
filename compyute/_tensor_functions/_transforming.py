"""Tensor transformation functions module"""

from typing import Optional

from .._types import _AxisLike, _ComplexLike, _DtypeLike, _ScalarLike, _ShapeLike
from ..engine import _get_engine
from ..tensors import Tensor, _as_tensor, tensor

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
    "tanh",
    "sech",
    "abs",
    "sqrt",
    "fft1d",
    "ifft1d",
    "fft2d",
    "ifft2d",
    "real",
]


def sum(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Sum of tensor elements over a given axis.

    Parameters
    ----------
    x: Tensor
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
    return _as_tensor(x.data.sum(axis=axis, keepdims=keepdims))


def prod(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Product of tensor elements over a given axis.

    Parameters
    ----------
    x: Tensor
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
    return _as_tensor(x.data.prod(axis=axis, keepdims=keepdims))


def mean(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Mean of tensor elements over a given axis.

    Parameters
    ----------
    x: Tensor
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
    return _as_tensor(x.data.mean(axis=axis, keepdims=keepdims))


def var(
    x: Tensor, axis: Optional[_AxisLike] = None, ddof: int = 0, keepdims: bool = False
) -> Tensor:
    """Variance of tensor elements over a given axis.

    Parameters
    ----------
    x: Tensor
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
    return _as_tensor(x.data.var(axis=axis, ddof=ddof, keepdims=keepdims))


def std(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Standard deviation of tensor elements over a given axis.

    Parameters
    ----------
    x: Tensor
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
    return _as_tensor(x.data.std(axis=axis, keepdims=keepdims))


def min(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Minimum of tensor elements over a given axis.

    Parameters
    ----------
    x: Tensor
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
    return _as_tensor(x.data.min(axis=axis, keepdims=keepdims))


def max(x: Tensor, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
    """Maximum of tensor elements over a given axis.

    Parameters
    ----------
    x: Tensor
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
    return _as_tensor(x.data.max(axis=axis, keepdims=keepdims))


def round(x: Tensor, decimals: int) -> Tensor:
    """Rounds the value of tensor elements.

    Parameters
    ----------
    x: Tensor
        Input tensor.
    decimals : int
        Decimal places of rounded values.

    Returns
    -------
    Tensor
        Tensor containing the rounded values.
    """
    return Tensor(x.data.round(decimals))


def exp(x: Tensor) -> Tensor:
    """Returns a new tensor containing the element-wise exponential values."""
    return Tensor(_get_engine(x.device).exp(x.data))


def log(x: Tensor) -> Tensor:
    """Returns a new tensor containing the element-wise log values."""
    return Tensor(_get_engine(x.device).log(x.data))


def log10(x: Tensor) -> Tensor:
    """Returns a new tensor containing the element-wise log base 10 values."""
    return Tensor(_get_engine(x.device).log10(x.data))


def log2(x: Tensor) -> Tensor:
    """Returns a new tensor containing the element-wise log base 2 values."""
    return Tensor(_get_engine(x.device).log2(x.data))


def sin(x: Tensor) -> Tensor:
    """Returns a new tensor containing the element-wise sine values."""
    return Tensor(_get_engine(x.device).sin(x.data))


def sinh(x: Tensor) -> Tensor:
    """Returns a new tensor containing the element-wise hyperbolic sine values."""
    return Tensor(_get_engine(x.device).sinh(x.data))


def cos(x: Tensor) -> Tensor:
    """Returns a new tensor containing the element-wise cosine values."""
    return Tensor(_get_engine(x.device).cos(x.data))


def cosh(x: Tensor) -> Tensor:
    """Returns a new tensor containing the element-wise hyperbolic cosine values."""
    return Tensor(_get_engine(x.device).cosh(x.data))


def tan(x: Tensor) -> Tensor:
    """Returns a new tensor containing the element-wise tangent values."""
    return Tensor(_get_engine(x.device).tan(x.data))


def tanh(x: Tensor) -> Tensor:
    """Returns a new tensor containing the element-wise hyperbolic tangent values."""
    return Tensor(_get_engine(x.device).tanh(x.data))


def sech(x: Tensor) -> Tensor:
    """Returns a new tensor containing the element-wise hyperbolic secant values."""
    return cosh(x) ** -1


def abs(x: Tensor) -> Tensor:
    """Returns a new tensor containing the absolute values."""
    return Tensor(_get_engine(x.device).abs(x.data))


def sqrt(x: Tensor) -> Tensor:
    """Returns a new tensor containing the square root values."""
    return Tensor(_get_engine(x.device).sqrt(x.data))


def fft1d(
    x: Tensor,
    n: Optional[int] = None,
    axis: int = -1,
    dtype: Optional[_ComplexLike] = None,
) -> Tensor:
    """Computes the 1D Fast Fourier Transform over a specified axis.

    Parameters
    ----------
    x: Tensor
        Input tensor.
    n : int, optional
        Length of the transformed axis of the output, by default None.
    axis : int, optional
        Axis over which to compute the FFT, by default -1.
    dtype : ComplexLike, optional
        Datatype of the output tensor, by default None.

    Returns
    -------
    Tensor
        Complex tensor.
    """
    return tensor(
        _get_engine(x.device).fft.fft(x.data, n=n, axis=axis), device=x.device, dtype=dtype
    )


def ifft1d(
    x: Tensor,
    n: Optional[int] = None,
    axis: int = -1,
    dtype: Optional[_ComplexLike] = None,
) -> Tensor:
    """Computes the inverse 1D Fast Fourier Transform over a specified axis.

    Parameters
    ----------
    x: Tensor
        Input tensor.
    n : int, optional
        Length of the transformed axis of the output, by default None.
    axis : int, optional
        Axis over which to compute the inverse FFT, by default -1.
    dtype : FloatLike, optional
        Datatype of the output tensor, by default None.

    Returns
    -------
    Tensor
        Float tensor.
    """
    return tensor(
        _get_engine(x.device).fft.ifft(x.data, n=n, axis=axis), device=x.device, dtype=dtype
    )


def fft2d(
    x: Tensor,
    s: Optional[_ShapeLike] = None,
    axes: tuple[int, int] = (-2, -1),
    dtype: Optional[_ComplexLike] = None,
) -> Tensor:
    """Computes the 2D Fast Fourier Transform over two specified axes.

    Parameters
    ----------
    x: Tensor
        Input tensor.
    n : ShapeLike, optional
        Shape (length of each transformed axis) of the output, by default None.
    axes : tuple[int, int], optional
        Axes over which to compute the FFT, by default (-2, -1).
    dtype : ComplexLike, optional
        Datatype of the output tensor, by default None.

    Returns
    -------
    Tensor
        Complex tensor.
    """
    return tensor(
        _get_engine(x.device).fft.fft2(x.data, s=s, axes=axes), device=x.device, dtype=dtype
    )


def ifft2d(
    x: Tensor,
    s: Optional[_ShapeLike] = None,
    axes: tuple[int, int] = (-2, -1),
    dtype: Optional[_ComplexLike] = None,
) -> Tensor:
    """Applies the inverse 1D Fast Fourier Transform to the tensor.

    Parameters
    ----------
    x: Tensor
        Input tensor.
    n : ShapeLike, optional
        Shape (length of each transformed axis) of the output, by default None.
    axes : tuple[int, int], optional
        Axes over which to compute the inverse FFT, by default (-2, -1).
    dtype : ComplexLike, optional
        Datatype of the output tensor, by default None.

    Returns
    -------
    Tensor
        Complex tensor.
    """
    return tensor(
        _get_engine(x.device).fft.ifft2(x.data, s=s, axes=axes), device=x.device, dtype=dtype
    )


def real(x: Tensor, dtype: Optional[_DtypeLike] = None) -> Tensor:
    """Returns the real parts of the complex tensor.

    Parameters
    ----------
    x: Tensor
        Input tensor.
    dtype : DtypeLike, optional
        Datatype of the output tensor, by default None.

    Returns
    -------
    Tensor
        Tensor containing real values.
    """
    return tensor(_get_engine(x.device).real(x.data), device=x.device, dtype=dtype)


def clip(x: Tensor, min_value: Optional[float] = None, max_value: Optional[float] = None) -> Tensor:
    """Limits the values of a tensor.

    Parameters
    ----------
    x: Tensor
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
    return Tensor(_get_engine(x.device).clip(x.data, min_value, max_value))
