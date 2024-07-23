"""Neural network functions module"""

from typing import Callable, Optional

from ...base_tensor import Tensor
from ...tensor_functions.computing import maximum, minimum
from ...tensor_functions.creating import identity
from ...tensor_functions.reshaping import insert_dim, reshape, tile
from ...tensor_functions.transforming import exp
from ...tensor_functions.transforming import max as cpmax
from ...tensor_functions.transforming import sech
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
    """Applies the Rectified Linear Unit function.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    return_grad_fn: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    y = maximum(x, 0)

    if return_grad_fn:
        return y, (lambda dy: (y > 0) * dy)
    return y, None


def leaky_relu(
    x: Tensor, alpha: float = 0.01, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the leaky ReLU function.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    alpha : float, optional
        Slope of the negative output, by default 0.01.
    return_grad_fn: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    x = x.float()
    y = maximum(x, 0) + alpha * minimum(x, 0)

    if return_grad_fn:
        return y, (lambda dy: ((y > 0).float() + (y < 0).float() * alpha) * dy)
    return y, None


def gelu(
    x: Tensor, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the Gaussian Error Linear Unit function.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    return_grad_fn: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
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
    """Applies the sigmoid function.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    return_grad_fn: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    y = exp(x) * (1 + exp(x)) ** -1

    if return_grad_fn:
        return y, (lambda dy: (y * (1 - y)) * dy)
    return y, None


def tanh(
    x: Tensor, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the hyperbolic tangent function.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    return_grad_fn: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    y = cptanh(x)

    if return_grad_fn:
        return y, (lambda dy: (1 - y**2) * dy)
    return y, None


def softmax(
    x: Tensor, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Applies the softmax function over the last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    return_grad_fn: bool, optional
        Whether to also return the according gradient function, by default False.

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
    """Applies the softmax function with temperature to the last axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    temperature : float, optional
        Temperature scaling to be applied in the calculation.

    Returns
    -------
    Tensor
        Output tensor.
    """
    if temperature == 0:
        raise ValueError("Temperature cannot be 0.")

    x = exp((x - cpmax(x, axis=-1, keepdims=True)) / temperature)
    return x / cpsum(x, axis=-1, keepdims=True)
