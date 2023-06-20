"""Normalization functions"""

import numpy as np
from numpynn import layers, inits, utils
from numpynn.tensor import Tensor


class Normalization(layers.ParamLayer):
    """Normalization layer base class"""


class Layernorm(Normalization):
    """Implements layer normalization."""

    def __init__(self, eps: float = 1e-7):
        super().__init__(None, None, None, True)
        self.eps = eps
        self.__var_inv = None
        self.__xhat = None
        self.g = Tensor()
        self.params = [self.b, self.g]

    def compile(self, i, prev_layer, succ_layer) -> None:
        super().compile(i, prev_layer, succ_layer)
        gamma = inits.ones((1, self.prev_layer.y.shape[1]))
        self.g.data = utils.expand_dims(gamma, self.prev_layer.y.ndim)
        self.b.data = inits.zeros_like(self.g.data)
        self.forward()

    def forward(self):
        super().forward()
        std_axis = tuple(i + 1 for i in range(self.x.ndim - 1))
        mean = np.mean(self.x.data, axis=std_axis, keepdims=True)
        var = np.var(self.x.data, axis=std_axis, keepdims=True, ddof=1)
        self.__var_inv = (var + self.eps)**-0.5
        self.__xhat = (self.x.data - mean) * self.__var_inv
        self.y.data = self.g.data * self.__xhat + self.b.data

    def backward(self):
        super().backward()
        sum_axis_bg = (0,) + tuple(i + 2 for i in range(self.x.ndim - 2))
        # gamma gradients
        self.g.grad = np.sum(self.__xhat * self.y.grad, axis=sum_axis_bg, keepdims=True)
        # beta gradients
        self.b.grad =  np.sum(self.y.grad, axis=sum_axis_bg, keepdims=True)
        # input gradients
        n = self.x.data[0].size
        sum_axis_x = tuple(i + 1 for i in range(self.x.ndim - 1))
        temp1 = n * self.y.grad
        temp2 = np.sum(self.y.grad, sum_axis_x, keepdims=True)
        temp3 = n / (n - 1) * self.__xhat * np.sum(self.y.grad * self.__xhat, sum_axis_x, keepdims=True)
        self.x.grad = self.g.data * self.__var_inv / n * (temp1 - temp2 - temp3)
