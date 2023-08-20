"""Activation layers module"""

import numpy as np

from walnut.tensor import Tensor, NumpyArray
from walnut.nn.funcional import sigmoid, softmax, relu
from walnut.nn.module import Module


__all__ = ["Relu", "Sigmoid", "Tanh", "Softmax"]


class Relu(Module):
    """Implements the ReLu activation function."""

    def __call__(self, x: Tensor) -> Tensor:
        y = relu(x)

        if self.training:

            def backward(y_grad: NumpyArray) -> NumpyArray:
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

            def backward(y_grad: NumpyArray) -> NumpyArray:
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

            def backward(y_grad: NumpyArray) -> NumpyArray:
                self.set_y_grad(y_grad)
                return y.data * (-y.data + 1.0) * y_grad

            self.backward = backward

        self.set_y(y)
        return y


class Softmax(Module):
    """Implements the Softmax activation function."""

    def __call__(self, x: Tensor) -> Tensor:
        y = softmax(x)

        if self.training:

            def backward(y_grad: NumpyArray) -> NumpyArray:
                self.set_y_grad(y_grad)
                channels = x.shape[-1]
                # credits to https://themaverickmeerkat.com/2019-10-23-Softmax/
                x1 = np.einsum("ij,ik->ijk", y.data, y.data)
                x2 = np.einsum("ij,jk->ijk", y.data, np.eye(channels, channels))
                return np.einsum("ijk,ik->ij", x2 - x1, y_grad)

            self.backward = backward

        self.set_y(y)
        return y


ACTIVATIONS = {
    "relu": Relu,
    "tanh": Tanh,
    "sigmoid": Sigmoid,
    "softmax": Softmax,
}
