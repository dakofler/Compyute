"""Activation layers module"""

from typing import Optional
from ..module import Module
from ...functional import gelu, leaky_relu, relu, sigmoid, tanh
from ....basetensor import Tensor


__all__ = ["ReLU", "LeakyReLU", "GELU", "Sigmoid", "Tanh"]


class ReLU(Module):
    """ReLu activation function."""

    def forward(self, x: Tensor) -> Tensor:
        y, self.backward_fn = relu(x, self.training)
        return y


class LeakyReLU(Module):
    """Leaky ReLu activation function."""

    def __init__(
        self,
        alpha: float = 0.01,
        label: Optional[str] = None,
    ):
        """Leaky ReLu activation function.

        Parameters
        ----------
        alpha : float, optional
            Slope of the negative output, by default 0.01.
        label: str, optional
            Module label.
        """
        super().__init__(label)
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        y, self.backward_fn = leaky_relu(x, self.alpha, self.training)
        return y


class GELU(Module):
    """Gaussian Error Linear Unit activation function."""

    def forward(self, x: Tensor) -> Tensor:
        y, self.backward_fn = gelu(x, self.training)
        return y


class Tanh(Module):
    """Tanh activation function."""

    def forward(self, x: Tensor) -> Tensor:
        y, self.backward_fn = tanh(x, self.training)
        return y


class Sigmoid(Module):
    """Sigmoid activation function."""

    def forward(self, x: Tensor) -> Tensor:
        y, self.backward_fn = sigmoid(x, self.training)
        return y


ACTIVATIONS = {
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "gelu": GELU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
}


def get_act_from_str(activation: str) -> Module:
    """Returns an instance of an actiation function."""
    if activation not in ACTIVATIONS.keys():
        raise ValueError(f"Unknown activation function {activation}.")
    return ACTIVATIONS[activation]()
