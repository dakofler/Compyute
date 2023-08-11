"""Activation layers module"""

import numpy as np

from walnut.tensor import Tensor, NumpyArray
import walnut.tensor_utils as tu
from walnut.nn.funcional import sigmoid, softmax
from walnut.nn.module import Module


__all__ = ["Relu", "Sigmoid", "Tanh", "Softmax"]


class Relu(Module):
    """Implements the ReLu activation function."""

    def __call__(self, x: Tensor) -> Tensor:
        y = tu.maximum(x, 0)

        if self.training:

            def backward(y_grad: NumpyArray) -> NumpyArray:
                x_grad = (y.data > 0) * y_grad
                self.set_y_grad(y_grad)
                return x_grad

            self.backward = backward

        self.set_y(y)
        return y


class Tanh(Module):
    """Implements the Tanh activation function."""

    def __call__(self, x: Tensor) -> Tensor:
        y = x.tanh()

        if self.training:

            def backward(y_grad: NumpyArray) -> NumpyArray:
                x_grad = (-y.data**2 + 1.0) * y_grad
                self.set_y_grad(y_grad)
                return x_grad

            self.backward = backward

        self.set_y(y)
        return y


class Sigmoid(Module):
    """Implements the Sigmoid activation function."""

    def __call__(self, x: Tensor) -> Tensor:
        y = sigmoid(x)

        if self.training:

            def backward(y_grad: NumpyArray) -> NumpyArray:
                x_grad = y.data * (-y.data + 1.0) * y_grad
                self.set_y_grad(y_grad)
                return x_grad

            self.backward = backward

        self.set_y(y)
        return y


class Softmax(Module):
    """Implements the Softmax activation function."""

    def __call__(self, x: Tensor) -> Tensor:
        y = softmax(x)

        if self.training:

            def backward(y_grad: NumpyArray) -> NumpyArray:
                channels = x.shape[-1]
                # credits to https://themaverickmeerkat.com/2019-10-23-Softmax/
                x1 = np.einsum("ij,ik->ijk", y.data, y.data)
                x2 = np.einsum("ij,jk->ijk", y.data, np.eye(channels, channels))
                x_grad = np.einsum("ijk,ik->ij", x2 - x1, y_grad)
                self.set_y_grad(y_grad)
                return x_grad

            self.backward = backward

        self.set_y(y)
        return y


ACTIVATIONS = {
    "relu": Relu,
    "tanh": Tanh,
    "sigmoid": Sigmoid,
    "softmax": Softmax,
}
