"""Activation layers module"""

from ..module import Module
from ...functional import sigmoid, relu, leaky_relu
from ....tensor import Tensor


__all__ = ["ReLU", "LeakyReLU", "GELU", "Sigmoid", "Tanh"]


class ReLU(Module):
    """Implements the ReLu activation function."""

    def forward(self, x: Tensor) -> Tensor:
        y = relu(x)

        if self.training:
            self.backward_fn = lambda dy: (y > 0) * dy

        return y


class LeakyReLU(Module):
    """Implements the Leaky ReLu activation function."""

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        y = leaky_relu(x, self.alpha)

        if self.training:
            self.backward_fn = (
                lambda dy: ((y > 0).float() + (y < 0).float() * self.alpha) * dy
            )

        return y


GELU_S = 0.7978845608028654  # sqrt(2/pi)
GELU_C = 0.044715


class GELU(Module):
    """Implements the Gaussian Error Linear Unit function."""

    def forward(self, x: Tensor) -> Tensor:
        tmp = GELU_S * (x + GELU_C * x**3)
        y = 0.5 * x * (1 + tmp.tanh())

        if self.training:
            self.backward_fn = (
                lambda dy: (
                    0.5 * (1 + tmp.tanh())
                    + 0.5 * x * tmp.sech() ** 2 * GELU_S * (1 + 3 * GELU_C * x**2)
                )
                * dy
            )

        return y


class Tanh(Module):
    """Implements the Tanh activation function."""

    def forward(self, x: Tensor) -> Tensor:
        y = x.tanh()

        if self.training:
            self.backward_fn = lambda dy: (1 - y**2) * dy

        return y


class Sigmoid(Module):
    """Implements the Sigmoid activation function."""

    def forward(self, x: Tensor) -> Tensor:
        y = sigmoid(x)

        if self.training:
            self.backward_fn = lambda dy: (y * (1 - y)) * dy

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
