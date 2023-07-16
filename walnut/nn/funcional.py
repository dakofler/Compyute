"""Functional module"""

import numpy as np
import numpy.fft as npfft

from walnut.tensor import Tensor, ShapeError


__all__ = ["sigmoid", "softmax", "convolve2d"]


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
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    mode: str = "valid",
) -> Tensor:
    """Convolves two tensors using their trailing two dims.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    stride : int, optional
        Stride used for the convolution operation, by default 1.
    dilation : int | tuple [int, int], optional
        Dilation used on filters, by default 1.
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

    match mode:
        case "full":
            p = (f.shape[-2] - 1, f.shape[-1] - 1)
        case "same":
            p = (f.shape[-2] // 2, f.shape[-1] // 2)
        case _:
            p = (0, 0)

    # padding
    d = x.ndim
    tpl = tuple((p[-d + ax], p[-d + ax]) if ax >= d - 2 else (0, 0) for ax in range(d))
    x = Tensor(np.pad(x.data, tpl))

    # convolution
    conv_shape = x.shape[-2:]
    x_fft = npfft.fft2(x.data, s=conv_shape).astype("complex64")
    f_fft = npfft.fft2(f.data, s=conv_shape).astype("complex64")
    ifft = Tensor(np.real(npfft.ifft2(x_fft * f_fft)))

    # slicing
    out_y = 1 + (x.shape[-2] - f.shape[-2])
    out_x = 1 + (x.shape[-1] - f.shape[-1])
    match x.ndim:
        case 2:
            return ifft[-out_y:, -out_x:][::stride, ::stride]
        case 3:
            return ifft[:, -out_y:, -out_x:][:, ::stride, ::stride]
        case 4:
            return ifft[:, :, -out_y:, -out_x:][:, :, ::stride, ::stride]
        case 5:
            return ifft[:, :, :, -out_y:, -out_x:][:, :, :, ::stride, ::stride]
        case _:
            return ifft
