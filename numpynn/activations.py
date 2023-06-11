# activation layers module

from numpynn import layers
import numpy as np


class Relu(layers.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.is_activation_layer = True

    def compile(self, id: int, prev_layer: object, succ_layer: object) -> None:
        super().compile(id, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = np.maximum(0, self.x)
    
    def backward(self) -> None:
        super().backward()
        self.dx = (self.y > 0).astype(int) * self.dy


class Sigmoid(layers.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.is_activation_layer = True

    def compile(self, id: int, prev_layer: object, succ_layer: object) -> None:
        super().compile(id, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = self.__sigmoid(self.x)
    
    def backward(self) -> None:
        super().backward()
        self.dx = self.__sigmoid(self.y) * (1.0 - self.__sigmoid(self.y)) * self.dy      

    def __sigmoid(self, v) -> np.ndarray:
        v = np.clip(v, -100, 100) # clipping because normalization is not implemented yet
        return 1.0 / (1.0 + np.exp(-v))


class Tanh(layers.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.is_activation_layer = True

    def compile(self, id: int, prev_layer: object, succ_layer: object) -> None:
        super().compile(id, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = np.tanh(self.x)
    
    def backward(self) -> None:
        super().backward()
        self.dx = (1.0 - self.y**2) * self.dy


class Softmax(layers.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.is_activation_layer = True

    def compile(self, id: int, prev_layer: object, succ_layer: object) -> None:
        super().compile(id, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        # will be improved later
        self.logits = self.x
        logit_maxes = np.amax(self.logits, axis=1, keepdims=True)
        self.norm_logits = self.logits - logit_maxes
        self.counts = np.exp(self.norm_logits)
        self.counts_sum = np.sum(self.counts, axis=1, keepdims=True)
        self.counts_sum_inv = self.counts_sum**-1
        probs = self.counts * self.counts_sum_inv
        self.y = probs
    
    def backward(self) -> None:
        super().backward()
        # will be improved later
        dcounts_sum_inv = np.sum(self.counts * self.dy, axis=1, keepdims=True)
        dcounts = self.counts_sum_inv * self.dy
        dcounts_sum = -self.counts_sum**-2 * dcounts_sum_inv
        dcounts += np.ones_like(self.counts) * dcounts_sum
        dnorm_logits = np.exp(self.norm_logits) * dcounts
        dlogits = dnorm_logits.copy()
        dlogit_maxes = np.sum(-dnorm_logits, axis=1, keepdims=True)
        dlogits += np.argmax(self.logits, axis=1, keepdims=True) * dlogit_maxes
        self.dx = dlogits