"""Activation layers module"""

from ..module import Module
from ...functional import gelu, leaky_relu, relu, sigmoid, tanh
from ....tensor import Tensor


__all__ = ["ReLU", "LeakyReLU", "GELU", "Sigmoid", "Tanh"]


class ReLU(Module):
    """Implements the ReLu activation function."""

    def forward(self, x: Tensor) -> Tensor:
        y, self.backward_fn = relu(x, self.training)
        return y


class LeakyReLU(Module):
    """Implements the Leaky ReLu activation function."""

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        y, self.backward_fn = leaky_relu(x, self.alpha, self.training)
        return y


class GELU(Module):
    """Implements the Gaussian Error Linear Unit function."""

    def forward(self, x: Tensor) -> Tensor:
        y, self.backward_fn = gelu(x, self.training)
        return y


class Tanh(Module):
    """Implements the Tanh activation function."""

    def forward(self, x: Tensor) -> Tensor:
        y, self.backward_fn = tanh(x, self.training)
        return y


class Sigmoid(Module):
    """Implements the Sigmoid activation function."""

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
