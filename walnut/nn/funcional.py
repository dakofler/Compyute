"""Functional module"""

import numpy as np
import numpy.fft as npfft
import cupy as cp
import cupy.fft as cpfft

from walnut.tensor import Tensor, ShapeError, ShapeLike
import walnut.tensor_utils as tu


__all__ = [
    "sigmoid",
    "softmax",
    "convolve1d",
    "convolve2d",
    "dilate1d",
    "dilate2d",
    "pad1d",
    "pad2d",
]


def relu(x: Tensor) -> Tensor:
    """Applies the ReLU function.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return tu.maximum(x, 0)


def sigmoid(x: Tensor) -> Tensor:
    """Applies the sigmoid function.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return ((-x.clip(-100, 100)).exp() + 1.0) ** -1 * 1.0


def softmax(x: Tensor) -> Tensor:
    """Applies the softmax function over the last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    expo = (x - x.max(axis=-1, keepdims=True)).exp()
    return expo / expo.sum(axis=-1, keepdims=True)


def convolve1d(
    x: Tensor,
    f: Tensor,
    stride: int = 1,
    dil: int = 1,
    pad: str | int | tuple[int, int] = "causal",
) -> Tensor:
    """Convolves two tensors along their trailing dim.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    stride : int, optional
        Stride used in the convolution operation, by default 1.
    dil : int, optional
        Dilation used in the filter, by default 1.
    pad : str | int, optional
        Padding applied before convolution.
        Options are "causal", "valid", "same" and "full" or the padding width as int,
        by default "causal".

    Returns
    -------
    Tensor
        Output tensor.

    Raises
    -------
    ShapeError
        If dimensions of input are < 3.
    ShapeError
        If dimensions of input and filter do not match.
    NotImplementedError
        If padding is "same" and the kernel shape is even.
    """
    if x.ndim != f.ndim:
        raise ShapeError("Dimensions of input and filter must match.")
    if pad == "same" and f.shape[-1] % 2 == 0 and dil == 1:
        raise NotImplementedError("Same padding and even kernel size not compatible.")

    f_dil = dilate1d(f, dil).data
    x_pad = pad1d(x, f_dil.shape, pad)

    # convolution
    if x.device == "cpu":
        ifft = np.real(
            npfft.ifft(
                npfft.fft(x_pad.data).astype("complex64")
                * npfft.fft(f_dil, n=x_pad.shape[-1]).astype("complex64")
            )
        )
    else:
        ifft = cp.real(
            cpfft.ifft(
                cpfft.fft(x_pad.data).astype("complex64")
                * cpfft.fft(f_dil, n=x_pad.shape[-1]).astype("complex64")
            )
        )

    # slicing
    out = 1 + (x_pad.shape[-1] - f_dil.shape[-1])
    slc_out = [slice(None)] * ifft.ndim
    slc_out[ifft.ndim - 1] = slice(-out, None)
    slc_stride = [slice(None)] * ifft.ndim
    slc_stride[ifft.ndim - 1] = slice(None, None, stride)
    return Tensor(ifft[*slc_out][*slc_stride], device=x.device)


def convolve2d(
    x: Tensor,
    f: Tensor,
    stride: int | tuple[int, int] = 1,
    dil: int | tuple[int, int] = 1,
    pad: str | tuple[int, int] = "valid",
) -> Tensor:
    """Convolves two tensors along their last two axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    stride : int | tuple [int, int], optional
        Strides used for each axis in the convolution operation, by default 1.
    dil : int | tuple [int, int], optional
        Dilations used for each axis of the filter, by default 1.
    pad : str | tuple[int, int], optional
        Padding applied before convolution.
        Options are "valid", "same" and "full" or a tuple of padding widths, by default "valid".

    Returns
    -------
    Tensor
        Output tensor.

    Raises
    -------
    ShapeError
        If dimensions of input are < 4.
    ShapeError
        If dimensions of input and filter do not match.
    NotImplementedError
        If padding is "same" and the kernel shape is even.
    """
    if x.ndim != f.ndim:
        raise ShapeError("Dimensions of input and filter must match.")
    if pad == "same" and f.shape[-1] % 2 == 0 and dil == 1:
        raise NotImplementedError("Same padding and even kernel size not compatible.")

    f_dil = dilate2d(f, dil).data
    x_pad = pad2d(x, f_dil.shape, pad)

    # convolution
    if x.device == "cpu":
        ifft = np.real(
            npfft.ifft2(
                npfft.fft2(x_pad.data).astype("complex64")
                * npfft.fft2(f_dil, s=x_pad.shape[-2:]).astype("complex64")
            )
        )
    else:
        ifft = cp.real(
            cpfft.ifft2(
                cpfft.fft2(x_pad.data).astype("complex64")
                * cpfft.fft2(f_dil, s=x_pad.shape[-2:]).astype("complex64")
            )
        )

    # slicing
    out_y = 1 + (x_pad.shape[-2] - f_dil.shape[-2])
    out_x = 1 + (x_pad.shape[-1] - f_dil.shape[-1])
    s_y, s_x = (stride, stride) if isinstance(stride, int) else stride
    slc_out = [slice(None)] * ifft.ndim
    slc_out[ifft.ndim - 2 :] = [slice(-out_y, None), slice(-out_x, None)]
    slc_stride = [slice(None)] * ifft.ndim
    slc_stride[ifft.ndim - 2 :] = [slice(None, None, s_y), slice(None, None, s_x)]
    return Tensor(ifft[*slc_out][*slc_stride], device=x.device)


def dilate1d(f: Tensor, dil: int) -> Tensor:
    """Dilates a tensor along axis -1.

    Parameters
    ----------
    x : Tensor
        Tensor to be dilated.
    dil : int
        Dilations used.

    Returns
    -------
    Tensor
        Dilated tensor.
    """
    if dil == 1:
        return f

    dim = f.ndim
    tpl = tuple(
        ((f.shape[d] - 1) * dil + 1) if d == dim - 1 else f.shape[d] for d in range(dim)
    )
    f_dil = tu.zeros(tpl, f.dtype, f.device)
    slc_dil = [slice(None)] * dim
    slc_dil[dim - 1] = slice(None, None, dil)
    f_dil[*slc_dil] = f
    return f_dil


def dilate2d(f: Tensor, dil: int | tuple[int, int]) -> Tensor:
    """Dilates a tensor along axis -2, -1.

    Parameters
    ----------
    x : Tensor
        Tensor to be dilated.
    dil : int | tuple [int, int]
        Dilations used for each axis of the tensor.

    Returns
    -------
    Tensor
        Dilated tensor.
    """
    dil = (dil, dil) if isinstance(dil, int) else dil
    if dil == (1, 1):
        return f

    dim = f.ndim
    tpl = tuple(
        ((f.shape[d] - 1) * dil[-dim + d] + 1) if d >= dim - 2 else f.shape[d]
        for d in range(dim)
    )
    f_dil = tu.zeros(tpl, f.dtype, f.device)
    slc_dil = [slice(None)] * dim
    slc_dil[dim - 2 :] = [slice(None, None, dil[0]), slice(None, None, dil[1])]
    f_dil[*slc_dil] = f
    return f_dil


def pad1d(
    x: Tensor, filter_shape: ShapeLike, pad: str | int | tuple[int, int]
) -> Tensor:
    """Pads axis -1 of a tensor.

    Parameters
    ----------
    x : Tensor
        Tensor to be padded.
    filter_shape : ShapeLike
        Shape of the filter tensor.
    pad : str | int | tuple[int, int]
        Padding applied before convolution.
        Options are "causal", "valid", "same" and "full" or the padding width as int.

    Returns
    -------
    Tensor
        Padded tensor.

    Raises
    -------
    NotImplementedError
        If padding type is invalid.
    """
    if (
        not isinstance(pad, int)
        and not isinstance(pad, tuple)
        and pad not in ("valid", "same", "full", "causal")
    ):
        raise NotImplementedError(f"Invalid padding type {pad}.")

    if isinstance(pad, int):
        p = (pad,) * 2
    elif isinstance(pad, tuple):
        p = pad
    else:
        match pad:
            case "full":
                p = (filter_shape[-1] - 1,) * 2
            case "same":
                p = (filter_shape[-1] // 2,) * 2
            case "causal":
                p = (filter_shape[-1] - 1, 0)
            case _:
                p = (0, 0)
    widths = tuple([(0, 0)] * (x.ndim - 1) + [p])
    return x.pad(widths)


def pad2d(x: Tensor, filter_shape: ShapeLike, pad: str | tuple[int, int]) -> Tensor:
    """Pads axis -2, -1 of a tensor.

    Parameters
    ----------
    x : Tensor
        Tensor to be padded.
    filter_shape : ShapeLike
        Shape of the filter tensor.
    pad : str | tuple[int, int]
        Padding applied to the tensor.
        Options are "valid", "same" and "full" or a tuple of padding widths.

    Returns
    -------
    Tensor
        Padded tensor.

    Raises
    -------
    NotImplementedError
        If padding type is invalid.
    """
    if not isinstance(pad, tuple) and pad not in ("valid", "same", "full"):
        raise NotImplementedError(f"Invalid padding type {pad}.")

    if isinstance(pad, tuple):
        p = pad
    else:
        match pad:
            case "full":
                p = (filter_shape[-2] - 1, filter_shape[-1] - 1)
            case "same":
                p = (filter_shape[-2] // 2, filter_shape[-1] // 2)
            case _:
                p = (0, 0)
    widths = tuple([(0, 0)] * (x.ndim - 2) + [(p[0], p[0]), (p[1], p[1])])
    return x.pad(widths)


def stretch2d(
    x: Tensor,
    stretches: tuple[int, int],
    shape: ShapeLike,
    axis: tuple[int, int] = (-2, -1),
) -> Tensor:
    """Stretches a tensor by repeating it's elements over given axis.

    Parameters
    ----------
    x : Tensor
        Tensor to be stretched out.
    stretches : tuple[int, int]
        Number of repeating values along each axis.
    shape : ShapeLike
        Shape of the target tensor. If the shape does not match after stretching,
        remaining values are filled with zeroes.
    axis : tuple[int, int], optional
        Axis along which to stretch the tensor, by default (-2, -1).

    Returns
    -------
    Tensor
        Stretched out tensor.
    """
    fa1, fa2 = stretches
    ax1, ax2 = axis
    x_stretched = x.repeat(fa1, ax1).repeat(fa2, ax2)
    return x_stretched.resize(shape)
