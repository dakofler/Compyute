"""Activation modules."""

from typing import Literal, Optional

from ...base_tensor import Tensor
from ..functional.activations import gelu, leaky_relu, relu, sigmoid, tanh
from .module import Module

__all__ = ["ReLU", "LeakyReLU", "GELU", "Sigmoid", "Tanh"]


class ReLU(Module):
    r"""Applies the Rectified Linear Unit activation function to an input tensor.

    .. math::
        y = \max(0, x)

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.
    """

    def forward(self, x: Tensor) -> Tensor:
        y, self._backward = relu(x, self._training)
        return y


class LeakyReLU(Module):
    r"""Leaky ReLu activation function.

    .. math::
        y = \max(\alpha \cdot x, x)

    Parameters
    ----------
    alpha : float, optional
        Slope of the negative output. Defaults to ``0.01``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.
    """

    def __init__(self, alpha: float = 0.01, label: Optional[str] = None, training: bool = False):
        super().__init__(label, training)
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        y, self._backward = leaky_relu(x, self.alpha, self._training)
        return y


class GELU(Module):
    r"""Gaussian Error Linear Unit activation function.

    .. math::
        y = 0.5 \cdot x \cdot (1 + \tanh(\sqrt{\frac{2}{pi}} \cdot (x + 0.044715 \cdot x^3)))

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.
    """

    def forward(self, x: Tensor) -> Tensor:
        y, self._backward = gelu(x, self._training)
        return y


class Tanh(Module):
    r"""Tanh activation function.

    .. math::
        y = tanh(x)

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.
    """

    def forward(self, x: Tensor) -> Tensor:
        y, self._backward = tanh(x, self._training)
        return y


class Sigmoid(Module):
    r"""Sigmoid activation function.

    .. math::
        y = \frac{1}{1 + e^{-x}}

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.
    """

    def forward(self, x: Tensor) -> Tensor:
        y, self._backward = sigmoid(x, self._training)
        return y


_ActivationLike = Literal["relu", "leaky_relu", "gelu", "sigmoid", "tanh"]
ACTIVATIONS = {
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "gelu": GELU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
}


def get_activation(activation: _ActivationLike) -> Module:
    """Returns an instance of an actiation function."""

    if activation not in ACTIVATIONS:
        raise ValueError(f"Unknown activation function: {activation}.")
    return ACTIVATIONS[activation]()
