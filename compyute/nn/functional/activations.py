"""Neural network activation functions."""

import math
from typing import Callable, Optional

from ...base_tensor import Tensor
from ...tensor_ops.creating import identity
from ...tensor_ops.reshaping import reshape, tile
from ...tensor_ops.transforming import exp
from ...tensor_ops.transforming import max as cpmax
from ...tensor_ops.transforming import maximum, sech
from ...tensor_ops.transforming import sum as cpsum
from ...tensor_ops.transforming import tanh as cptanh

__all__ = [
    "relu",
    "leaky_relu",
    "gelu",
    "sigmoid",
    "silu",
    "tanh",
    "softmax",
    "temperature_softmax",
]


def relu(
    x: Tensor, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the Rectified Linear Unit activation function to an input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    ----------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor], optional
        Gradient function.

    See Also
    --------
    :class:`compyute.nn.ReLU`
    """
    y = maximum(x, 0)

    if return_grad_fn:
        return y, (lambda dy: (y > 0) * dy)
    return y, None


def leaky_relu(
    x: Tensor, alpha: float = 0.01, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the leaky ReLU function to an input tensor as.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    alpha : float, optional
        Slope of the negative output. Defaults to ``0.01``.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    ----------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor], optional
        Gradient function.

    See Also
    --------
    :class:`compyute.nn.LeakyReLU`
    """
    y = maximum(alpha * x, x)

    if return_grad_fn:

        def grad_fn(dy: Tensor) -> Tensor:
            return ((y >= 0) + (y < 0).to_type(x.dtype) * alpha) * dy

        return y, grad_fn

    return y, None


def gelu(
    x: Tensor, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the Gaussian Error Linear Unit function to an input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.

    See Also
    --------
    :class:`compyute.nn.GELU`
    """

    tmp = math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
    y = 0.5 * x * (1 + cptanh(tmp))

    if return_grad_fn:

        def grad_fn(dy: Tensor) -> Tensor:
            dx1 = 1 + cptanh(tmp)
            dx2 = x * sech(tmp) ** 2 * math.sqrt(2 / math.pi) * (1 + 0.13415 * x**2)
            return 0.5 * (dx1 + dx2) * dy

        return y, grad_fn

    return y, None


def sigmoid(
    x: Tensor, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the sigmoid function to an input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.

    See Also
    --------
    :class:`compyute.nn.Sigmoid`
    """
    x_exp = exp(x)
    y = x_exp / (1 + x_exp)

    if return_grad_fn:
        return y, (lambda dy: (y * (1 - y)) * dy)

    return y, None


def silu(
    x: Tensor, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the Sigmoid Linear Unit activation function to an input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.

    See Also
    --------
    :class:`compyute.nn.SiLU`
    """
    sig, sigmoid_grad_fn = sigmoid(x, return_grad_fn=return_grad_fn)
    y = sig * x

    if return_grad_fn and sigmoid_grad_fn is not None:
        return y, (lambda dy: x * sigmoid_grad_fn(dy) + sig * dy)

    return y, None


def tanh(
    x: Tensor, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the hyperbolic tangent activationfunction to an input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.

    See Also
    --------
    :class:`compyute.nn.Tanh`
    """
    y = cptanh(x)

    if return_grad_fn:
        return y, (lambda dy: (1 - y**2) * dy)

    return y, None


def softmax(
    x_exp: Tensor, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the softmax function over the last axis of an input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.

    See Also
    --------
    :class:`compyute.nn.Softmax`
    """
    x_exp = exp(x_exp - cpmax(x_exp, axis=-1, keepdims=True))
    y = x_exp / cpsum(x_exp, axis=-1, keepdims=True)

    if return_grad_fn:

        def grad_fn(dy: Tensor) -> Tensor:
            dx = tile(y.to_shape((*y.shape, 1)), y.shape[-1], -1)
            dx *= identity(y.shape[-1], device=y.device) - dx.T
            return reshape(dx @ dy.to_shape((*dy.shape, 1)), y.shape)

        return y, grad_fn
    return y, None


def temperature_softmax(x: Tensor, temperature: float = 1) -> Tensor:
    r"""Applies the softmax function with temperature over the last axis of an input tensor.

    .. math::
        y = \frac{e^{\frac{x}{T}}}{\sum_{i=1}^N e^{\frac{x_i}{T}}}

    Parameters
    ----------
    x : Tensor
        Input tensor.
    temperature : float, optional
        Temperature scaling to be applied in the calculation. Defaults to ``1``.

    Returns
    -------
    Tensor
        Output tensor.

    Raises
    ------
    ValueError
        If temperature is 0.
    """
    if temperature == 0:
        raise ValueError("Temperature cannot be 0.")

    x_exp = exp((x - cpmax(x, axis=-1, keepdims=True)) / temperature)
    return x_exp / cpsum(x_exp, axis=-1, keepdims=True)
