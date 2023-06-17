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
        self.y = np.maximum(0, self.x)

    def backward(self) -> None:
        super().backward()
        self.dx = (self.y > 0) * self.dy


class Sigmoid(Activation):
    """Implements the Sigmoid activation function."""

    def compile(self, i, prev_layer, succ_layer):
        super().compile(i, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = self.__sigmoid(self.x)

    def backward(self) -> None:
        super().backward()
        sigm = self.__sigmoid(self.y)
        self.dx = sigm * (1.0 - sigm) * self.dy

    def __sigmoid(self, v):
        # clip to avoid high values when exponentiating
        v = np.clip(v, -100, 100)
        return 1.0 / (1.0 + np.exp(-v))


class Tanh(Activation):
    """Implements the Tanh activation function."""

    def compile(self, i, prev_layer, succ_layer):
        super().compile(i, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = np.tanh(self.x)

    def backward(self) -> None:
        super().backward()
        self.dx = (1.0 - self.y**2) * self.dy


class Softmax(Activation):
    """Implements the Softmax activation function."""

    def compile(self, i, prev_layer, succ_layer):
        super().compile(i, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = self.__softmax(self.x)

    def backward(self) -> None:
        super().backward()
        _, x = self.x.shape
        # credits to https://themaverickmeerkat.com/2019-10-23-Softmax/
        tensor1 = np.einsum('ij,ik->ijk', self.y, self.y)
        tensor2 = np.einsum('ij,jk->ijk', self.y, np.eye(x, x))
        delta = tensor2 - tensor1
        self.dx = np.einsum('ijk,ik->ij', delta, self.dy)

    def __softmax(self, tensor, axis=1):
        # subtract max value to avoid high values when exponentiating.
        e = np.exp(tensor - np.amax(tensor, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)
