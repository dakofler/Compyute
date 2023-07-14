"""Activation modules module"""

import numpy as np

from walnut.tensor import Tensor, NumpyArray
from walnut.nn.modules.utility import Module


class Relu(Module):
    """Implements the ReLu activation function."""

    def __call__(self, x: Tensor) -> Tensor:
        super().__call__(x)
        self.y.data = np.maximum(0, self.x.data)
        return self.y

    def backward(self, y_grad: NumpyArray) -> NumpyArray:
        super().backward(y_grad)
        self.x.grad = (self.y.data > 0) * self.y.grad
        return self.x.grad


class Sigmoid(Module):
    """Implements the Sigmoid activation function."""

    def __call__(self, x: Tensor) -> Tensor:
        super().__call__(x)
        self.y.data = self.__sigmoid(self.x.data)
        return self.y

    def backward(self, y_grad: NumpyArray) -> NumpyArray:
        super().backward(y_grad)
        sigm = self.__sigmoid(self.y.data)
        self.x.grad = sigm * (1.0 - sigm) * self.y.grad
        return self.x.grad

    def __sigmoid(self, x: NumpyArray):
        x = np.clip(x, -100, 100)  # clip to avoid high values when exponentiating
        return 1.0 / (1.0 + np.exp(-x))


class Tanh(Module):
    """Implements the Tanh activation function."""

    def __call__(self, x: Tensor) -> Tensor:
        super().__call__(x)
        self.y.data = np.tanh(self.x.data)
        return self.y

    def backward(self, y_grad: NumpyArray) -> NumpyArray:
        super().backward(y_grad)
        self.x.grad = (1.0 - self.y.data**2) * self.y.grad
        return self.x.grad


class Softmax(Module):
    """Implements the Softmax activation function."""

    def __call__(self, x: Tensor) -> Tensor:
        super().__call__(x)
        self.y.data = self.__softmax(self.x.data)
        return self.y

    def backward(self, y_grad: NumpyArray) -> NumpyArray:
        super().backward(y_grad)
        _, channels = self.x.shape
        # credits to https://themaverickmeerkat.com/2019-10-23-Softmax/
        x1 = np.einsum("ij,ik->ijk", self.y.data, self.y.data)
        x2 = np.einsum("ij,jk->ijk", self.y.data, np.eye(channels, channels))
        self.x.grad = np.einsum("ijk,ik->ij", x2 - x1, self.y.grad)
        return self.x.grad

    def __softmax(self, x: NumpyArray) -> NumpyArray:
        expo = np.exp(x - np.amax(x, axis=1, keepdims=True))
        return expo / np.sum(expo, axis=1, keepdims=True)


ACTIVATIONS = {
    "relu": Relu,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
}
