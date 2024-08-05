"""Neural network activation functions."""

from typing import Callable, Optional

from ...base_tensor import Tensor
from ...tensor_functions.creating import identity
from ...tensor_functions.reshaping import insert_dim, reshape, tile
from ...tensor_functions.transforming import exp
from ...tensor_functions.transforming import max as cpmax
from ...tensor_functions.transforming import maximum, minimum, sech
from ...tensor_functions.transforming import sum as cpsum
from ...tensor_functions.transforming import tanh as cptanh

__all__ = [
    "relu",
    "leaky_relu",
    "gelu",
    "sigmoid",
    "tanh",
    "softmax",
    "temperature_softmax",
]
_GELU_S: float = 0.7978845608028654
_GELU_C: float = 0.044715


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
    x = x.to_float()
    y = maximum(alpha * x, x)

    if return_grad_fn:
        return y, (lambda dy: ((y > 0).to_float() + (y < 0).to_float() * alpha) * dy)
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

    tmp = _GELU_S * (x + _GELU_C * x**3)
    y = 0.5 * x * (1 + cptanh(tmp))

    if return_grad_fn:
        return y, (
            lambda dy: (
                0.5 * (1 + cptanh(tmp))
                + 0.5 * x * sech(tmp) ** 2 * _GELU_S * (1 + 3 * _GELU_C * x**2)
            )
            * dy
        )
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
    y = exp(x) * (1 + exp(x)) ** -1

    if return_grad_fn:
        return y, (lambda dy: (y * (1 - y)) * dy)
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
    x: Tensor, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    r"""Applies the softmax function over the last axis of an input tensor.

    .. math::
        softmax(x) = \frac{\exp(x)}{\sum_{i=1}^N \exp(x_i)}

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
    """
    x = exp(x - cpmax(x, axis=-1, keepdims=True))
    y = x / cpsum(x, axis=-1, keepdims=True)

    if return_grad_fn:

        def grad_fn(dy: Tensor) -> Tensor:
            sm_ = tile(insert_dim(y, -1), y.shape[-1], -1)
            return reshape(
                sm_ * (identity(y.shape[-1], device=x.device) - sm_.T) @ insert_dim(dy, -1), y.shape
            )

        return y, grad_fn
    return y, None


def temperature_softmax(x: Tensor, temperature: float = 1) -> Tensor:
    r"""Applies the softmax function with temperature over the last axis of an input tensor.

    .. math::
        softmax(x) = \frac{\exp(\frac{x}{T})}{\sum_{i=1}^N \exp(\frac{x_i}{T})}

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

    x = exp((x - cpmax(x, axis=-1, keepdims=True)) / temperature)
    return x / cpsum(x, axis=-1, keepdims=True)
