"""Functional module"""

import numpy as np
import numpy.fft as npfft

from walnut.tensor import Tensor


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


def convolve2d(x: Tensor, f: Tensor, stride: int = 1) -> Tensor:
    """Convolves two tensors using their trainling two dims.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    stride : int, optional
        Strides used for the convolution, by default 1.

    Returns
    -------
    Tensor
        Output tensor.
    """
    target_shape = x.shape[-2:]
    x_fft = npfft.fft2(x.data, s=target_shape).astype("complex64")
    f_fft = npfft.fft2(f.data, s=target_shape).astype("complex64")
    return Tensor(np.real(npfft.ifft2(x_fft * f_fft)).astype("float32"))
