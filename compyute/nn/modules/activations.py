"""Activation layers module"""

from typing import Literal, Optional

from ...base_tensor import Tensor
from ..functional.activations import gelu, leaky_relu, relu, sigmoid, tanh
from .module import Module

__all__ = ["ReLU", "LeakyReLU", "GELU", "Sigmoid", "Tanh"]


class ReLU(Module):
    """ReLu activation function."""

    def forward(self, x: Tensor) -> Tensor:
        y, self._backward = relu(x, self._training)
        return y


class LeakyReLU(Module):
    """Leaky ReLu activation function."""

    def __init__(self, alpha: float = 0.01, label: Optional[str] = None, training: bool = False):
        """Leaky ReLu activation function.

        Parameters
        ----------
        alpha : float, optional
            Slope of the negative output, by default 0.01.
        label: str, optional
            Module label.
        training: bool, optional
            Whether the module should be in training mode, by default False.
        """
        super().__init__(label, training)
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        y, self._backward = leaky_relu(x, self.alpha, self._training)
        return y


class GELU(Module):
    """Gaussian Error Linear Unit activation function."""

    def forward(self, x: Tensor) -> Tensor:
        y, self._backward = gelu(x, self._training)
        return y


class Tanh(Module):
    """Tanh activation function."""

    def forward(self, x: Tensor) -> Tensor:
        y, self._backward = tanh(x, self._training)
        return y


class Sigmoid(Module):
    """Sigmoid activation function."""

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
