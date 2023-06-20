"""activation layers module"""

import numpy as np
from numpynn.layers import Layer


class Activation(Layer):
    """Actiation layer base class"""


class Relu(Activation):
    """Implements the ReLu activation function."""

    def compile(self, i, prev_layer, succ_layer):
        super().compile(i, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y.data = np.maximum(0, self.x.data)

    def backward(self) -> None:
        super().backward()
        self.x.grad = (self.y.data > 0) * self.y.grad


class Sigmoid(Activation):
    """Implements the Sigmoid activation function."""

    def compile(self, i, prev_layer, succ_layer):
        super().compile(i, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y.data = self.__sigmoid(self.x.data)

    def backward(self) -> None:
        super().backward()
        sigm = self.__sigmoid(self.y.data)
        self.x.grad = sigm * (1.0 - sigm) * self.y.grad

    def __sigmoid(self, tensor):
        # clip to avoid high values when exponentiating
        tensor = np.clip(tensor, -100, 100)
        return 1.0 / (1.0 + np.exp(-tensor))


class Tanh(Activation):
    """Implements the Tanh activation function."""

    def compile(self, i, prev_layer, succ_layer):
        super().compile(i, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y.data = np.tanh(self.x.data)

    def backward(self) -> None:
        super().backward()
        self.x.grad = (1.0 - self.y.data**2) * self.y.grad


class Softmax(Activation):
    """Implements the Softmax activation function."""

    def compile(self, i, prev_layer, succ_layer):
        super().compile(i, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y.data = self.__softmax(self.x.data)

    def backward(self) -> None:
        super().backward()
        _, channels = self.x.shape
        # credits to https://themaverickmeerkat.com/2019-10-23-Softmax/
        tensor1 = np.einsum('ij,ik->ijk', self.y.data, self.y.data)
        tensor2 = np.einsum('ij,jk->ijk', self.y.data, np.eye(channels, channels))
        delta = tensor2 - tensor1
        self.x.grad = np.einsum('ijk,ik->ij', delta, self.y.grad)

    def __softmax(self, tensor, axis=1):
        # subtract max value to avoid high values when exponentiating.
        expo = np.exp(tensor - np.amax(tensor, axis=axis, keepdims=True))
        return expo / np.sum(expo, axis=axis, keepdims=True)
