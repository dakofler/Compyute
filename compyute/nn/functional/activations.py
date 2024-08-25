"""Neural network activation functions."""

import math

from ...tensor_ops.creating import identity
from ...tensor_ops.reshaping import insert_dim, reshape, tile
from ...tensor_ops.transforming import exp
from ...tensor_ops.transforming import max as cpmax
from ...tensor_ops.transforming import maximum, sech
from ...tensor_ops.transforming import sum as cpsum
from ...tensor_ops.transforming import tanh as cptanh
from ...tensors import Tensor
from .functions import Function, FunctionCache, PseudoCache

__all__ = ["relu", "leaky_relu", "gelu", "sigmoid", "silu", "tanh", "softmax"]


class FReLU(Function):
    """Applies the softmax function over the last axis of an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        y = maximum(x, 0)
        cache.relu_y = y
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        return (cache.relu_y > 0) * dy


def relu(x: Tensor) -> Tensor:
    """Applies the Rectified Linear Unit activation function to an input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    ----------
    Tensor
        Output tensor.

    See Also
    --------
    :class:`compyute.nn.ReLU`
    """
    return FReLU.forward(PseudoCache(), x)


class FLeakyReLU(Function):
    """Applies the leaky ReLU function to an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, alpha: float) -> Tensor:
        y = maximum(alpha * x, x)
        cache.leaky_relu_y = y
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor, alpha: float) -> Tensor:
        return (
            (cache.leaky_relu_y >= 0)
            + (cache.leaky_relu_y < 0).to_type(dy.dtype) * alpha
        ) * dy


def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    """Applies the leaky ReLU function to an input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    alpha : float, optional
        Slope of the negative output. Defaults to ``0.01``.

    Returns
    ----------
    Tensor
        Output tensor.

    See Also
    --------
    :class:`compyute.nn.LeakyReLU`
    """
    return FLeakyReLU.forward(PseudoCache(), x, alpha)


class FGELU(Function):
    """Applies the Gaussian Error Linear Unit function to an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        tmp = math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
        y = 0.5 * x * (1 + cptanh(tmp))
        cache.gelu_x, cache.gelu_tmp = x, tmp
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, tmp = cache.gelu_x, cache.gelu_tmp

        dx1 = 1 + cptanh(tmp)
        dx2 = x * sech(tmp) ** 2 * math.sqrt(2 / math.pi) * (1 + 0.13415 * x**2)
        return 0.5 * (dx1 + dx2) * dy


def gelu(x: Tensor) -> Tensor:
    """Applies the Gaussian Error Linear Unit function to an input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    --------
    :class:`compyute.nn.GELU`
    """
    return FGELU.forward(PseudoCache(), x)


class FSigmoid(Function):
    """Applies the sigmoid function to an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        x_exp = exp(x)
        y = x_exp / (1 + x_exp)
        cache.sigmoid_y = y
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        return (cache.sigmoid_y * (1 - cache.sigmoid_y)) * dy


def sigmoid(x: Tensor) -> Tensor:
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
    Callable, optional
        Gradient function.

    See Also
    --------
    :class:`compyute.nn.Sigmoid`
    """
    return FSigmoid.forward(PseudoCache(), x)


class FSiLU(Function):
    """Applies the Sigmoid Linear Unit activation function to an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        cache.silu_x = x  # FSigmoid already caches y
        return x * FSigmoid.forward(cache, x)

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        return cache.silu_x * FSigmoid.backward(cache, dy) + cache.sigmoid_y * dy


def silu(x: Tensor) -> Tensor:
    """Applies the Sigmoid Linear Unit activation function to an input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    --------
    :class:`compyute.nn.SiLU`
    """
    return FSiLU.forward(PseudoCache(), x)


class FSoftmax(Function):
    """Applies the softmax function over the last axis of an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        x = exp(x - cpmax(x, axis=-1, keepdims=True))
        y = x / cpsum(x, axis=-1, keepdims=True)
        cache.softmax_y = y
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        dx = tile(insert_dim(cache.softmax_y, -1), dy.shape[-1], -1)
        dx *= identity(dy.shape[-1], device=dy.device) - dx.T
        return reshape(dx @ dy.to_shape((*dy.shape, 1)), dy.shape)


def softmax(x: Tensor) -> Tensor:
    """Applies the softmax function over the last axis of an input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    --------
    :class:`compyute.nn.Softmax`
    """
    return FSoftmax.forward(PseudoCache(), x)


class FTanh(Function):
    """Applies the hyperbolic tangent activationfunction to an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        y = cptanh(x)
        cache.tanh_y = y
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        return (1 - cache.tanh_y**2) * dy


def tanh(x: Tensor) -> Tensor:
    """Applies the hyperbolic tangent activationfunction to an input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    --------
    :class:`compyute.nn.Tanh`
    """
    return FTanh.forward(PseudoCache(), x)
