"""Functional module"""

import numpy as np
import numpy.fft as npfft

from walnut.tensor import Tensor, ShapeError


__all__ = ["sigmoid", "softmax", "convolve2d", "dilate"]


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
    """Applies the softmax function.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    expo = np.exp(x.data - np.amax(x.data, axis=1, keepdims=True))
    return Tensor(expo / np.sum(expo, axis=1, keepdims=True))


def convolve2d(
    x: Tensor,
    f: Tensor,
    strides: int | tuple[int, int] = 1,
    dil: int | tuple[int, int] = 1,
    mode: str = "valid",
) -> Tensor:
    """Convolves two tensors using their trailing two dims.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    strides : int | tuple [int, int], optional
        Strides used for each axis in the convolution operation, by default 1.
    dil : int | tuple [int, int], optional
        Dilations used for each axis of the filter, by default 1.
    mode : str, optional
        Convolution mode, by default "valid".

    Returns
    -------
    Tensor
        Output tensor.

    Raises
    -------
    ShapeError
        If dimensions of input and filter do not match.
    """
    if x.ndim != f.ndim:
        raise ShapeError("Dimensions of input and filter must match.")

    # dilation
    f_dil = dilate(f, dil)

    # padding
    match mode:
        case "full":
            p = (f_dil.shape[-2] - 1, f_dil.shape[-1] - 1)
        case "same":
            p = (f_dil.shape[-2] // 2, f_dil.shape[-1] // 2)
        case _:
            p = (0, 0)
    d = x.ndim
    tpl = tuple((p[-d + ax], p[-d + ax]) if ax >= d - 2 else (0, 0) for ax in range(d))
    x = Tensor(np.pad(x.data, tpl))

    # convolution
    conv_shape = x.shape[-2:]
    x_fft = npfft.fft2(x.data, s=conv_shape).astype("complex64")
    f_fft = npfft.fft2(f_dil.data, s=conv_shape).astype("complex64")
    ifft = Tensor(np.real(npfft.ifft2(x_fft * f_fft)))

    # slicing
    out_y = 1 + (x.shape[-2] - f_dil.shape[-2])
    out_x = 1 + (x.shape[-1] - f_dil.shape[-1])
    s_y, s_x = (strides, strides) if isinstance(strides, int) else strides
    match x.ndim:
        case 2:
            return ifft[-out_y:, -out_x:][::s_y, ::s_x]
        case 3:
            return ifft[:, -out_y:, -out_x:][:, ::s_y, ::s_x]
        case 4:
            return ifft[:, :, -out_y:, -out_x:][:, :, ::s_y, ::s_x]
        case 5:
            return ifft[:, :, :, -out_y:, -out_x:][:, :, :, ::s_y, ::s_x]
        case _:
            return ifft


def dilate(f: Tensor, dil: int | tuple[int, int]) -> Tensor:
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
