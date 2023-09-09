"""Activation layers module"""

from walnut.tensor import Tensor, ArrayLike
from walnut.nn.funcional import sigmoid, relu
from walnut.nn.module import Module


__all__ = ["ReLU", "Sigmoid", "Tanh"]


class ReLU(Module):
    """Implements the ReLu activation function."""

    def __call__(self, x: Tensor) -> Tensor:
        y = relu(x)

        if self.training:

            def backward(y_grad: ArrayLike) -> ArrayLike:
                self.set_y_grad(y_grad)
                return (y.data > 0) * y_grad

            self.backward = backward

        self.set_y(y)
        return y


class Tanh(Module):
    """Implements the Tanh activation function."""

    def __call__(self, x: Tensor) -> Tensor:
        y = x.tanh()

        if self.training:

            def backward(y_grad: ArrayLike) -> ArrayLike:
                self.set_y_grad(y_grad)
                return (-y.data**2 + 1.0) * y_grad

            self.backward = backward

        self.set_y(y)
        return y


class Sigmoid(Module):
    """Implements the Sigmoid activation function."""

    def __call__(self, x: Tensor) -> Tensor:
        y = sigmoid(x)

        if self.training:

            def backward(y_grad: ArrayLike) -> ArrayLike:
                self.set_y_grad(y_grad)
                return y.data * (1.0 - y.data) * y_grad

            self.backward = backward

        self.set_y(y)
        return y
