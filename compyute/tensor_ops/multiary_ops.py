"""Tensor multinary operations."""

from ..tensors import ShapeError, Tensor
from .unary_ops import fft1d, fft2d, ifft1d, ifft2d, real

__all__ = [
    "allclose",
    "convolve1d_fft",
    "convolve2d_fft",
    "dot",
    "einsum",
    "inner",
    "outer",
]


def allclose(x1: Tensor, x2: Tensor, *, rtol=1e-05, atol=1e-08) -> bool:
    """Returns ``True`` if all elements in the tensor are ``True``.

    Parameters
    ----------
    x1 : Tensor
        Input tensor.
    x2 : Tensor
        Input tensor.
    rtol : float
        Relative tolerance. Defaults to ``1e-05``.
    atol : float
        Absolute tolerance. Defaults to ``1e-08``.

    Returns
    -------
    bool
        ``True`` if all elements in the tensors are within the given tolerance.
    """
    return x1.device.module.allclose(x1.data, x2.data, rtol, atol)


def convolve1d_fft(x1: Tensor, x2: Tensor) -> Tensor:
    """Computes the convolution of two tensors using FFT over their last axis.

    Parameters
    ----------
    x1 : Tensor
        First tensor.
    x2 : Tensor
        Second tensor.

    Returns
    -------
    Tensor
        Convolution of the two tensors.
    """
    conv = real(ifft1d(fft1d(x1) * fft1d(x2, n=x1.shape[-1])))
    out = x1.shape[-1] - x2.shape[-1] + 1
    return conv[..., -out:].to_type(x1.dtype)


def convolve2d_fft(x1: Tensor, x2: Tensor) -> Tensor:
    """Computes the convolution of two tensors using FFT over their last two axes.

    Parameters
    ----------
    x1 : Tensor
        First tensor.
    x2 : Tensor
        Second tensor.

    Returns
    -------
    Tensor
        Convolution of the two tensors.
    """
    conv = real(ifft2d(fft2d(x1) * fft2d(x2, n=x1.shape[-2:])))
    out_y = x1.shape[-2] - x2.shape[-2] + 1
    out_x = x1.shape[-1] - x2.shape[-1] + 1
    return conv[..., -out_y:, -out_x:].to_type(x1.dtype)


def dot(x1: Tensor, x2: Tensor) -> Tensor:
    """Computes the dot product of two tensors.

    Parameters
    ----------
    x1 : Tensor
        First tensor.
    x2 : Tensor
        Second tensor.

    Returns
    -------
    Tensor
        Dot product of the tensors.
    """
    if x1.n_axes != 1 or x2.n_axes != 1:
        raise ShapeError("Inputs must be 1D-tensors.")
    return inner(x1, x2)


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
    data = tensors[0].device.module.einsum(subscripts, *[t.data for t in tensors])
    return Tensor(data)


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
    return Tensor(tensors[0].device.module.inner(*[t.data for t in tensors]))


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
    return Tensor(tensors[0].device.module.outer(*[t.data for t in tensors]))
