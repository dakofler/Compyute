"""Activation layers module"""

from compyute.nn.funcional import sigmoid, relu, leaky_relu
from compyute.nn.module import Module
from compyute.tensor import Tensor
from compyute.types import ArrayLike


__all__ = ["ReLU", "LeakyReLU", "GELU", "Sigmoid", "Tanh"]
GELU_S = 0.7978845608028654  # sqrt(2/pi)
GELU_C = 0.044715


class ReLU(Module):
    """Implements the ReLu activation function."""

    def forward(self, x: Tensor) -> Tensor:
        y = relu(x)

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                return (y > 0).data * dy

            self.backward = backward

        self.set_y(y)
        return y


class LeakyReLU(Module):
    """Implements the Leaky ReLu activation function."""

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        y = leaky_relu(x)

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                return ((y > 0).float() + (y < 0).float() * self.alpha).data * dy

            self.backward = backward

        self.set_y(y)
        return y


class GELU(Module):
    """Implements the Gaussian Error Linear Unit function."""

    def forward(self, x: Tensor) -> Tensor:
        tmp = GELU_S * (x + GELU_C * x**3)
        y = 0.5 * x * (1 + tmp.tanh())

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                return (
                    0.5 * (1 + tmp.tanh())
                    + 0.5 * x * tmp.sech() ** 2 * GELU_S * (1 + 3 * GELU_C * x**2)
                ).data * dy

            self.backward = backward

        self.set_y(y)
        return y


class Tanh(Module):
    """Implements the Tanh activation function."""

    def forward(self, x: Tensor) -> Tensor:
        y = x.tanh()

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                return (1 - y**2).data * dy

            self.backward = backward

        self.set_y(y)
        return y


class Sigmoid(Module):
    """Implements the Sigmoid activation function."""

    def forward(self, x: Tensor) -> Tensor:
        y = sigmoid(x)

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                return (y * (1 - y)).data * dy

            self.backward = backward

        self.set_y(y)
        return y
