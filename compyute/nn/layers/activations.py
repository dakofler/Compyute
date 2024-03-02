"""Activation layers module"""

from compyute.tensor import Tensor, ArrayLike
from compyute.nn.funcional import sigmoid, relu, leaky_relu
from compyute.nn.module import Module


__all__ = ["ReLU", "LeakyReLU", "Sigmoid", "Tanh"]


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


class Tanh(Module):
    """Implements the Tanh activation function."""

    def forward(self, x: Tensor) -> Tensor:
        y = x.tanh()

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                return (1.0 - y**2).data * dy

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
                return (y * (1.0 - y)).data * dy

            self.backward = backward

        self.set_y(y)
        return y
