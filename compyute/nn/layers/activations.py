"""Activation layers module"""

from compyute.tensor import Tensor, ArrayLike
from compyute.nn.funcional import sigmoid, relu
from compyute.nn.module import Module


__all__ = ["ReLU", "Sigmoid", "Tanh"]


class ReLU(Module):
    """Implements the ReLu activation function."""

    def __call__(self, x: Tensor) -> Tensor:
        y = relu(x)

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                return (y > 0).data * dy

            self.backward = backward

        self.set_y(y)
        return y


class Tanh(Module):
    """Implements the Tanh activation function."""

    def __call__(self, x: Tensor) -> Tensor:
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

    def __call__(self, x: Tensor) -> Tensor:
        y = sigmoid(x)

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                return (y * (1.0 - y)).data * dy

            self.backward = backward

        self.set_y(y)
        return y
