"""Neural network activation functions."""

import math

from ...tensor_ops.creating import identity
from ...tensor_ops.reshaping import insert_dim, reshape, tile
from ...tensor_ops.transforming import exp, invert
from ...tensor_ops.transforming import max as cp_max
from ...tensor_ops.transforming import maximum, sech
from ...tensor_ops.transforming import sum as cp_sum
from ...tensor_ops.transforming import tanh as cp_tanh
from ...tensors import Tensor
from .functions import Function, FunctionCache, PseudoCache

__all__ = ["relu", "leaky_relu", "gelu", "sigmoid", "silu", "tanh", "softmax"]


class FReLU(Function):
    """Applies the softmax function over the last axis of an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        y = maximum(x, 0)
        cache.y = y > 0
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        y = cache.y
        return y * dy


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
        cache.y = y > 0
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor, alpha: float) -> Tensor:
        y = cache.y
        return (y + invert(y).to_type(dy.dtype) * alpha) * dy


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
        y = 0.5 * x * (1 + cp_tanh(tmp))
        cache.x, cache.tmp = x, tmp
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, tmp = cache.x, cache.tmp
        dx1 = 1 + cp_tanh(tmp)
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
        cache.y = y
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        y = cache.y
        return (y * (1 - y)) * dy


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
        sig = FSigmoid.forward(cache, x)
        y = x * sig
        cache.x, cache.sig = x, sig
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, sig = cache.x, cache.sig
        return x * FSigmoid.backward(cache, dy) + sig * dy


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
        x = exp(x - cp_max(x, axis=-1, keepdims=True))
        y = x / cp_sum(x, axis=-1, keepdims=True)
        cache.y = y
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        y = cache.y
        dx = tile(insert_dim(y, -1), dy.shape[-1], -1)
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
        y = cp_tanh(x)
        cache.y = y
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        y = cache.y
        return (1 - y**2) * dy


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
