"""Activation modules."""

from typing import Literal, Optional, TypeAlias

from ...tensors import Tensor
from ..functional.activation_funcs import (
    FastGELUFn,
    GELUFn,
    LeakyReLUFn,
    ReLUFn,
    SigmoidFn,
    SiLUFn,
    SoftmaxFn,
    TanhFn,
)
from .module import Module

__all__ = [
    "ReLU",
    "LeakyReLU",
    "GELU",
    "FastGELU",
    "Sigmoid",
    "SiLU",
    "Softmax",
    "Tanh",
]


class GELU(Module):
    r"""Gaussian Error Linear Unit activation function (using the :math:`tanh` approximation).

    .. math::
        y = 0.5 \cdot x \cdot \left( 1 + \text{tanh} \left( x \sqrt{\frac{2}{\pi}} \cdot (1 + 0.044715 \cdot x^2) \right) \right)

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def forward(self, x: Tensor) -> Tensor:
        return GELUFn.forward(self.fcache, x)

    def backward(self, dy: Tensor) -> Tensor:
        return GELUFn.backward(self.fcache, dy)


class FastGELU(Module):
    r"""Gaussian Error Linear Unit activation function (using the :math:`sigmoid` approximation).

    .. math::
        y = x \cdot \sigma{1.702x}

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def forward(self, x: Tensor) -> Tensor:
        return FastGELUFn.forward(self.fcache, x)

    def backward(self, dy: Tensor) -> Tensor:
        return FastGELUFn.backward(self.fcache, dy)


class LeakyReLU(Module):
    r"""Leaky ReLu activation function.

    .. math::
        y = \text{max}(\alpha \cdot x, x)

    Parameters
    ----------
    alpha : float, optional
        Slope of the negative output. Defaults to ``0.01``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(self, alpha: float = 0.01, label: Optional[str] = None):
        super().__init__(label)
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return LeakyReLUFn.forward(self.fcache, x, self.alpha)

    def backward(self, dy: Tensor) -> Tensor:
        return LeakyReLUFn.backward(self.fcache, dy)


class ReLU(Module):
    r"""Applies the Rectified Linear Unit activation function to an input tensor.

    .. math::
        y = \text{max}(0, x)

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.

    """

    def forward(self, x: Tensor) -> Tensor:
        return ReLUFn.forward(self.fcache, x)

    def backward(self, dy: Tensor) -> Tensor:
        return ReLUFn.backward(self.fcache, dy)


class Sigmoid(Module):
    r"""Sigmoid activation function.

    .. math::
        y = \frac{1}{1 + e^{-x}}

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def forward(self, x: Tensor) -> Tensor:
        return SigmoidFn.forward(self.fcache, x)

    def backward(self, dy: Tensor) -> Tensor:
        return SigmoidFn.backward(self.fcache, dy)


class SiLU(Module):
    r"""Sigmoid Linear Unit activation function.

    .. math::
        y = x \cdot \text{sigmoid}(x) = \frac{x}{1 + e^{-x}}

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def forward(self, x: Tensor) -> Tensor:
        return SiLUFn.forward(self.fcache, x)

    def backward(self, dy: Tensor) -> Tensor:
        return SiLUFn.backward(self.fcache, dy)


class Softmax(Module):
    r"""Softmax activation function.

    .. math::
        y = \frac{e^x}{\sum_{i=1}^N e^{x_i}}

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def forward(self, x: Tensor) -> Tensor:
        return SoftmaxFn.forward(self.fcache, x)

    def backward(self, dy: Tensor) -> Tensor:
        return SoftmaxFn.backward(self.fcache, dy)


class Tanh(Module):
    r"""Tanh activation function.

    .. math::
        y = \text{tanh}(x)

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def forward(self, x: Tensor) -> Tensor:
        return TanhFn.forward(self.fcache, x)

    def backward(self, dy: Tensor) -> Tensor:
        return TanhFn.backward(self.fcache, dy)


ActivationLike: TypeAlias = Literal[
    "relu", "leaky_relu", "gelu", "sigmoid", "silu", "tanh"
]
ACTIVATIONS = {
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "gelu": GELU,
    "sigmoid": Sigmoid,
    "silu": SiLU,
    "tanh": Tanh,
}


def get_activation(activation: ActivationLike) -> Module:
    """Returns an actiation function."""

    if activation not in ACTIVATIONS:
        raise ValueError(f"Unknown activation function: {activation}.")
    return ACTIVATIONS[activation]()
