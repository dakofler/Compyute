"""Functional module"""

import numpy as np
import numpy.fft as npfft

from walnut.tensor import Tensor, ShapeError


__all__ = ["sigmoid", "softmax", "convolve1d", "convolve2d", "dilate1d", "dilate2d"]


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
    x_clip = np.clip(x.data, -100, 100)  # clip to avoid high values when exponentiating
    return Tensor(1.0 / (1.0 + np.exp(-x_clip)))


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
    expo = np.exp(x.data - np.amax(x.data, axis=-1, keepdims=True))
    return Tensor(expo / np.sum(expo, axis=-1, keepdims=True))


def convolve1d(
    x: Tensor,
    f: Tensor,
    stride: int = 1,
    dil: int = 1,
    mode: str | int = "same",
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
    mode : str | int, optional
        Padding applied before convolution.
        Options are "valid", "same" and "full" or the padding width as int, by default "same".

    Returns
    -------
    Tensor
        Output tensor.

    Raises
    -------
    ShapeError
        If dimensions of input and filter do not match.
    NotImplementedError
        If padding is "same" and the kernel shape is even.
    """
    if x.ndim < 3:
        raise ShapeError("Expected 3D input or higher.")
    if x.ndim != f.ndim:
        raise ShapeError("Dimensions of input and filter must match.")
    if mode == "same" and f.shape[-1] % 2 == 0 and dil == 1:
        raise NotImplementedError("Same padding and even kernel size not compatible.")

    # dilation
    f_dil = dilate1d(f, dil)

    # padding
    if isinstance(mode, int):
        p = mode
    else:
        match mode:
            case "full":
                p = f_dil.shape[-1] - 1
            case "same":
                p = f_dil.shape[-1] // 2
            case _:
                p = 0
    dim = x.ndim
    tpl = tuple((p, p) if d == dim - 1 else (0, 0) for d in range(dim))
    x_pad = Tensor(np.pad(x.data, tpl))

    # convolution
    conv_shape = x_pad.shape[-1]
    x_fft = npfft.fft(x_pad.data, n=conv_shape).astype("complex64")
    f_fft = npfft.fft(f_dil.data, n=conv_shape).astype("complex64")
    ifft = Tensor(np.real(npfft.ifft(x_fft * f_fft)))

    # slicing
    out = 1 + (x_pad.shape[-1] - f_dil.shape[-1])
    match x_pad.ndim:
        case 3:
            return ifft[:, :, -out:][:, :, ::stride]
        case 4:
            return ifft[:, :, :, -out:][:, :, :, ::stride]
        case 5:
            return ifft[:, :, :, :, -out:][:, :, :, :, ::stride]
        case _:
            return ifft


def convolve2d(
    x: Tensor,
    f: Tensor,
    stride: int | tuple[int, int] = 1,
    dil: int | tuple[int, int] = 1,
    mode: str | tuple[int, int] = "valid",
) -> Tensor:
    """Convolves two tensors along their last two trailing dims.

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
    mode : str | tuple[int, int], optional
        Padding applied before convolution.
        Options are "valid", "same" and "full" or a tuple of padding widths, by default "valid".

    Returns
    -------
    Tensor
        Output tensor.

    Raises
    -------
    ShapeError
        If dimensions of input and filter do not match.
    NotImplementedError
        If padding is "same" and the kernel shape is even.
    """
    if x.ndim < 4:
        raise ShapeError("Expected 4D input or higher.")
    if x.ndim != f.ndim:
        raise ShapeError("Dimensions of input and filter must match.")
    if mode == "same" and f.shape[-1] % 2 == 0 and dil == 1:
        raise NotImplementedError("Same padding and even kernel size not compatible.")

    # dilation
    f_dil = dilate2d(f, dil)

    # padding
    if isinstance(mode, tuple):
        p = mode
    else:
        match mode:
            case "full":
                p = (f_dil.shape[-2] - 1, f_dil.shape[-1] - 1)
            case "same":
                p = (f_dil.shape[-2] // 2, f_dil.shape[-1] // 2)
            case _:
                p = (0, 0)
    dim = x.ndim
    tpl = tuple(
        (p[-dim + d], p[-dim + d]) if d >= dim - 2 else (0, 0) for d in range(dim)
    )
    x = Tensor(np.pad(x.data, tpl))

    # convolution
    conv_shape = x.shape[-2:]
    x_fft = npfft.fft2(x.data, s=conv_shape).astype("complex64")
    f_fft = npfft.fft2(f_dil.data, s=conv_shape).astype("complex64")
    ifft = Tensor(np.real(npfft.ifft2(x_fft * f_fft)))

    # slicing
    out_y = 1 + (x.shape[-2] - f_dil.shape[-2])
    out_x = 1 + (x.shape[-1] - f_dil.shape[-1])
    s_y, s_x = (stride, stride) if isinstance(stride, int) else stride
    match x.ndim:
        case 4:
            return ifft[:, :, -out_y:, -out_x:][:, :, ::s_y, ::s_x]
        case 5:
            return ifft[:, :, :, -out_y:, -out_x:][:, :, :, ::s_y, ::s_x]
        case _:
            return ifft


def dilate1d(f: Tensor, dil: int) -> Tensor:
    """Dilates a tensor.

    Parameters
    ----------
    x : Tensor
        _description_
    dil : int | tuple [int, int]
        Dilations used for each axis of the tensor.

    Returns
    -------
    Tensor
        Dilated tensor.
    """
    dim = f.ndim
    tpl = tuple(
        ((f.shape[d] - 1) * dil + 1) if d == dim - 1 else f.shape[d] for d in range(dim)
    )
    f_dil = np.zeros(tpl)
    match f.ndim:
        case 3:
            f_dil[:, :, ::dil] = f.data
        case 4:
            f_dil[:, :, :, ::dil] = f.data
    return Tensor(f_dil)


def dilate2d(f: Tensor, dil: int | tuple[int, int]) -> Tensor:
    """Dilates a tensor.

    Parameters
    ----------
    x : Tensor
        _description_
    dil : int | tuple [int, int]
        Dilations used for each axis of the tensor.

    Returns
    -------
    Tensor
        Dilated tensor.
    """
    dim = f.ndim
    dil = (dil, dil) if isinstance(dil, int) else dil
    tpl = tuple(
        ((f.shape[d] - 1) * dil[-dim + d] + 1) if d >= dim - 2 else f.shape[d]
        for d in range(dim)
    )
    f_dil = np.zeros(tpl)
    match f.ndim:
        case 2:
            f_dil[:: dil[0], :: dil[1]] = f.data
        case 3:
            f_dil[:, :: dil[0], :: dil[1]] = f.data
        case 4:
            f_dil[:, :, :: dil[0], :: dil[1]] = f.data
        case 5:
            f_dil[:, :, :, :: dil[0], :: dil[1]] = f.data
    return Tensor(f_dil)
