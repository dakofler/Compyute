"""Activation modules."""

from typing import Literal, Optional, TypeAlias

from ...tensors import Tensor
from ..functional.activations import (
    FGELU,
    FLeakyReLU,
    FReLU,
    FSigmoid,
    FSiLU,
    FSoftmax,
    FTanh,
)
from .module import Module

__all__ = ["ReLU", "LeakyReLU", "GELU", "Sigmoid", "SiLU", "Softmax", "Tanh"]


class GELU(Module):
    r"""Gaussian Error Linear Unit activation function (using the :math:`tanh` approximation).

    .. math::
        y = 0.5 \cdot x \cdot (1 + \text{tanh}(\sqrt{\frac{2}{pi}} \cdot (x + 0.044715 \cdot x^3)))

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return FGELU.forward(self.fcache, x)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return FGELU.backward(self.fcache, dy)


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

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return FLeakyReLU.forward(self.fcache, x, self.alpha)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return FLeakyReLU.backward(self.fcache, dy)


class ReLU(Module):
    r"""Applies the Rectified Linear Unit activation function to an input tensor.

    .. math::
        y = \text{max}(0, x)

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.

    """

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return FReLU.forward(self.fcache, x)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return FReLU.backward(self.fcache, dy)


class Sigmoid(Module):
    r"""Sigmoid activation function.

    .. math::
        y = \frac{1}{1 + e^{-x}}

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return FSigmoid.forward(self.fcache, x)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return FSigmoid.backward(self.fcache, dy)


class SiLU(Module):
    r"""Sigmoid Linear Unit activation function.

    .. math::
        y = x \cdot \text{sigmoid}(x) = \frac{x}{1 + e^{-x}}

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return FSiLU.forward(self.fcache, x)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return FSiLU.backward(self.fcache, dy)


class Softmax(Module):
    r"""Softmax activation function.

    .. math::
        y = \frac{e^x}{\sum_{i=1}^N e^{x_i}}

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return FSoftmax.forward(self.fcache, x)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return FSoftmax.backward(self.fcache, dy)


class Tanh(Module):
    r"""Tanh activation function.

    .. math::
        y = \text{tanh}(x)

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return FTanh.forward(self.fcache, x)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return FTanh.backward(self.fcache, dy)


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
