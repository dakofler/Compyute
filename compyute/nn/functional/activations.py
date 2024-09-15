"""Neural network activation functions."""

from ...tensor_ops.creating import identity
from ...tensor_ops.reshaping import insert_dim, reshape, tile
from ...tensor_ops.selecting import maximum
from ...tensor_ops.transforming import exp, invert, sech
from ...tensor_ops.transforming import tanh as cp_tanh
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
        cache.alpha, cache.y = alpha, y > 0.0
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        alpha, y = cache.alpha, cache.y
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


class FSigmoid(Function):
    """Applies the sigmoid function to an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        x_exp = exp(x)
        y = x_exp / (1.0 + x_exp)
        cache.y = y
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        y = cache.y
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
    return FSigmoid.forward(PseudoCache(), x)


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
    return FTanh.forward(PseudoCache(), x)


class FGELU(Function):
    """Applies the Gaussian Error Linear Unit function to an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        tanh_out = cp_tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x))
        y = 0.5 * x * (1.0 + tanh_out)
        cache.x, cache.tanh_out = x, tanh_out
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, tanh_out = cache.x, cache.tanh_out
        dx1 = 1.0 + tanh_out
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
    return FGELU.forward(PseudoCache(), x)


class FFastGELU(Function):
    """Applies the Gaussian Error Linear Unit function to an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        sig = FSigmoid.forward(cache, 1.702 * x)
        y = x * sig
        cache.x, cache.sig = x, sig
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, sig = cache.x, cache.sig
        return dy * sig + x * FSigmoid.backward(cache, dy) * 1.702


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
    return FFastGELU.forward(PseudoCache(), x)


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
        x = exp(x - x.max(-1, keepdims=True))
        y = x / x.sum(-1, keepdims=True)
        cache.y = y
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        y = cache.y
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
    return FSoftmax.forward(PseudoCache(), x)
