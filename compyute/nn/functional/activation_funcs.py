"""Neural network activation functions."""

from ...tensor_ops.selection_ops import maximum
from ...tensor_ops.unary_ops import exp
from ...tensor_ops.unary_ops import tanh as _tanh
from ...tensors import Tensor
from ...typing import int8
from .functions import Function, FunctionCache, PseudoCache

__all__ = [
    "relu",
    "leaky_relu",
    "gelu",
    "fast_gelu",
    "sigmoid",
    "silu",
    "tanh",
    "softmax",
]


class ReLUFn(Function):
    """Applies the Rectified Linear Unit activation function to the input."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        y = maximum(x, 0.0)
        cache.push(y > 0.0)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        (mask,) = cache.pop()
        return dy * mask


def relu(x: Tensor) -> Tensor:
    """Applies the Rectified Linear Unit activation function to the input.

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
    return ReLUFn.forward(PseudoCache(), x)


class LeakyReLUFn(Function):
    """Applies the leaky ReLU function to an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, alpha: float) -> Tensor:
        y = maximum(alpha * x, x)
        cache.push(alpha, y > 0.0)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        alpha, mask = cache.pop()
        return dy * (mask + (~mask).to_type(dy.dtype) * alpha)


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
    return LeakyReLUFn.forward(PseudoCache(), x, alpha)


class SigmoidFn(Function):
    """Applies the sigmoid function to an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        y = 1.0 / (1.0 + exp(-x))
        cache.push(y)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        (y,) = cache.pop()
        return y * (1.0 - y) * dy


def sigmoid(x: Tensor) -> Tensor:
    """Applies the sigmoid function to an input tensor.

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
    :class:`compyute.nn.Sigmoid`
    """
    return SigmoidFn.forward(PseudoCache(), x)


class TanhFn(Function):
    """Applies the hyperbolic tangent activation function to the input."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        y = _tanh(x)
        cache.push(y)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        (y,) = cache.pop()
        return (1.0 - y * y) * dy


def tanh(x: Tensor) -> Tensor:
    """Applies the hyperbolic tangent activation function to the input.

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
    return TanhFn.forward(PseudoCache(), x)


class GELUFn(Function):
    """Applies the Gaussian Error Linear Unit activation function to the input."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        # sqrt(2/pi) = 0.7978845608
        tanh_term = _tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x))
        y = 0.5 * x * (1.0 + tanh_term)
        cache.push(x, tanh_term)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, tanh_term = cache.pop()
        dx1 = 1.0 + tanh_term
        # sqrt(2/pi) * 3 * 0.044715 = 0.1070322243
        dx2 = x * (1.0 - tanh_term * tanh_term) * (0.7978845608 + 0.1070322243 * x * x)
        return 0.5 * dy * (dx1 + dx2)


def gelu(x: Tensor) -> Tensor:
    """Applies the Gaussian Error Linear Unit activation function to the input.

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
    return GELUFn.forward(PseudoCache(), x)


class FastGELUFn(Function):
    """Applies the Gaussian Error Linear Unit activation function to the input."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        sig = 1.0 / (1.0 + exp(x * -1.702))
        y = x * sig
        cache.push(x, sig)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, sig = cache.pop()
        return dy * sig * (1.0 + x * 1.702 * (1.0 - sig))


def fast_gelu(x: Tensor) -> Tensor:
    """Applies the Gaussian Error Linear Unit activation function to the input.

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
    :class:`compyute.nn.FastGELU`
    """
    return FastGELUFn.forward(PseudoCache(), x)


class SiLUFn(Function):
    """Applies the Sigmoid Linear Unit activation function to the input."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        sig = 1.0 / (1.0 + exp(-x))
        y = x * sig
        cache.push(x, sig)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, sig = cache.pop()
        return dy * sig * (1.0 + x * (1.0 - sig))


def silu(x: Tensor) -> Tensor:
    """Applies the Sigmoid Linear Unit activation function to the input.

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
    return SiLUFn.forward(PseudoCache(), x)


class SoftmaxFn(Function):
    """Applies the softmax activation function to the last dimension of an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        x = exp(x - x.max(-1, keepdims=True))
        y = x / x.sum(-1, keepdims=True)
        cache.push(y)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        (y,) = cache.pop()
        return y * (dy - (dy * y).sum(-1, keepdims=True))  # thank you ChatGPT


def softmax(x: Tensor) -> Tensor:
    """Applies the softmax activation function to the last dimension of an input tensor.

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
    return SoftmaxFn.forward(PseudoCache(), x)
