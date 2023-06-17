# activation layers module

from numpynn.layers import Layer
import numpy as np


class Activation(Layer):

    def __init__(self) -> None:
        super().__init__()
        self.is_activation_layer = True

    def compile(self, id, prev_layer, succ_layer):
        super().compile(id, prev_layer, succ_layer)

    def forward(self) -> None:
        super().forward()
    
    def backward(self) -> None:
        super().backward()


class Relu(Activation):

    def __init__(self) -> None:
        super().__init__()

    def compile(self, id, prev_layer, succ_layer):
        super().compile(id, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = np.maximum(0, self.x)
    
    def backward(self) -> None:
        super().backward()
        self.dx = (self.y > 0) * self.dy


class Sigmoid(Activation):

    def __init__(self) -> None:
        super().__init__()

    def compile(self, id, prev_layer, succ_layer):
        super().compile(id, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = self.__sigmoid(self.x)
    
    def backward(self) -> None:
        super().backward()
        self.dx = self.__sigmoid(self.y) * (1.0 - self.__sigmoid(self.y)) * self.dy      

    def __sigmoid(self, v):
        # clip to avoid high values when exponentiating
        v = np.clip(v, -100, 100)
        return 1.0 / (1.0 + np.exp(-v))


class Tanh(Activation):

    def __init__(self) -> None:
        super().__init__()

    def compile(self, id, prev_layer, succ_layer):
        super().compile(id, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = np.tanh(self.x)
    
    def backward(self) -> None:
        super().backward()
        self.dx = (1.0 - self.y**2) * self.dy


class Softmax(Activation):
    
    def __init__(self) -> None:
        super().__init__()

    def compile(self, id, prev_layer, succ_layer):
        super().compile(id, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = self.__softmax(self.x)

    def backward(self) -> None:
        super().backward()
        _, x1 = self.x.shape
        # credits to https://themaverickmeerkat.com/2019-10-23-Softmax/
        tensor1 = np.einsum('ij,ik->ijk', self.y, self.y) 
        tensor2 = np.einsum('ij,jk->ijk', self.y, np.eye(x1, x1))
        dSoftmax = tensor2 - tensor1
        self.dx = np.einsum('ijk,ik->ij', dSoftmax, self.dy)

    def __softmax(self, array, axis=1):
        # subtract max value to avoid high values when exponentiating.
        e = np.exp(array - np.amax(array, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)
