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

    def forward(self, x: Tensor) -> Tensor:
        return FGELU.forward(self._fcache, x)

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        return FGELU.backward(self._fcache, dy)


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
        return FLeakyReLU.forward(self._fcache, x, self.alpha)

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        return FLeakyReLU.backward(self._fcache, dy, self.alpha)


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
        return FReLU.forward(self._fcache, x)

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        return FReLU.backward(self._fcache, dy)


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
        return FSigmoid.forward(self._fcache, x)

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        return FSigmoid.backward(self._fcache, dy)


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
        return FSiLU.forward(self._fcache, x)

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        return FSiLU.backward(self._fcache, dy)


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
        return FSoftmax.forward(self._fcache, x)

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        return FSoftmax.backward(self._fcache, dy)


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
        return FTanh.forward(self._fcache, x)

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        return FTanh.backward(self._fcache, dy)


_ActivationLike: TypeAlias = Literal[
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


def get_activation(activation: _ActivationLike) -> Module:
    """Returns an actiation function."""

    if activation not in ACTIVATIONS:
        raise ValueError(f"Unknown activation function: {activation}.")
    return ACTIVATIONS[activation]()
