"""Activation layers module"""

import numpy as np

from walnut.tensor import NumpyArray
from walnut.nn.layers.utility import Layer


class Relu(Layer):
    """Implements the ReLu activation function."""

    def forward(self, mode: str = "eval") -> None:
        self.y.data = np.maximum(0, self.x.data)

    def backward(self) -> None:
        self.x.grad = (self.y.data > 0) * self.y.grad


class Sigmoid(Layer):
    """Implements the Sigmoid activation function."""

    def forward(self, mode: str = "eval") -> None:
        self.y.data = self.__sigmoid(self.x.data)

    def backward(self) -> None:
        sigm = self.__sigmoid(self.y.data)
        self.x.grad = sigm * (1.0 - sigm) * self.y.grad

    def __sigmoid(self, x: NumpyArray):
        x = np.clip(x, -100, 100)  # clip to avoid high values when exponentiating
        return 1.0 / (1.0 + np.exp(-x))


class Tanh(Layer):
    """Implements the Tanh activation function."""

    def forward(self, mode: str = "eval") -> None:
        self.y.data = np.tanh(self.x.data)

    def backward(self) -> None:
        super().backward()
        self.x.grad = (1.0 - self.y.data**2) * self.y.grad


class Softmax(Layer):
    """Implements the Softmax activation function."""

    def forward(self, mode: str = "eval") -> None:
        self.y.data = self.__softmax(self.x.data)

    def backward(self) -> None:
        _, channels = self.x.shape
        # credits to https://themaverickmeerkat.com/2019-10-23-Softmax/
        x1 = np.einsum("ij,ik->ijk", self.y.data, self.y.data)
        x2 = np.einsum("ij,jk->ijk", self.y.data, np.eye(channels, channels))
        self.x.grad = np.einsum("ijk,ik->ij", x2 - x1, self.y.grad)

    def __softmax(self, x: NumpyArray) -> NumpyArray:
        expo = np.exp(x - np.amax(x, axis=1, keepdims=True))
        return expo / np.sum(expo, axis=1, keepdims=True)


ACTIVATIONS = {
    "relu": Relu,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
}
