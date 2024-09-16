"""Neural network activation functions."""

from ...tensor_ops.creating import identity
from ...tensor_ops.reshaping import insert_dim, reshape, tile
from ...tensor_ops.selecting import maximum
from ...tensor_ops.unary import exp, invert
from ...tensor_ops.unary import tanh as cp_tanh
from ...tensors import Tensor
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
    """Applies the softmax function over the last axis of an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        y = maximum(x, 0)
        cache.push(y > 0)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        (y,) = cache.pop()
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
        alpha, y = cache.pop()
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
    return LeakyReLUFn.forward(PseudoCache(), x, alpha)


class SigmoidFn(Function):
    """Applies the sigmoid function to an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        x_exp = exp(x)
        y = x_exp / (1.0 + x_exp)
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
    """Applies the hyperbolic tangent activationfunction to an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        y = cp_tanh(x)
        cache.push(y)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        (y,) = cache.pop()
        return (1.0 - y * y) * dy


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
    return TanhFn.forward(PseudoCache(), x)


class GELUFn(Function):
    """Applies the Gaussian Error Linear Unit function to an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        # sqrt(2/pi) = 0.7978845608
        tanh_out = cp_tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x))
        y = 0.5 * x * (1.0 + tanh_out)
        cache.push(x, tanh_out)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, tanh_out = cache.pop()
        dx1 = 1.0 + tanh_out
        # sqrt(2/pi) * 3 * 0.044715 = 0.1070322243
        dx2 = x * (1.0 - tanh_out * tanh_out) * (0.7978845608 + 0.1070322243 * x * x)
        return 0.5 * dy * (dx1 + dx2)


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
    return GELUFn.forward(PseudoCache(), x)


class FastGELUFn(Function):
    """Applies the Gaussian Error Linear Unit function to an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        sig = SigmoidFn.forward(cache, 1.702 * x)
        y = x * sig
        cache.push(x, sig)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, sig = cache.pop()
        return dy * sig + x * SigmoidFn.backward(cache, dy) * 1.702


def fast_gelu(x: Tensor) -> Tensor:
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
    :class:`compyute.nn.FastGELU`
    """
    return FastGELUFn.forward(PseudoCache(), x)


class SiLUFn(Function):
    """Applies the Sigmoid Linear Unit activation function to an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        sig = SigmoidFn.forward(cache, x)
        y = x * sig
        cache.push(x, sig)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, sig = cache.pop()
        return x * SigmoidFn.backward(cache, dy) + sig * dy


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
    return SiLUFn.forward(PseudoCache(), x)


class SoftmaxFn(Function):
    """Applies the softmax function over the last axis of an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        x = exp(x - x.max(-1, keepdims=True))
        y = x / x.sum(-1, keepdims=True)
        cache.push(y)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        (y,) = cache.pop()
        dx = tile(insert_dim(y, -1), dy.shape[-1], -1)
        dx *= identity(dy.shape[-1], device=dy.device) - dx.T
        return reshape(dx @ insert_dim(dy, -1), dy.shape)


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
    return SoftmaxFn.forward(PseudoCache(), x)
