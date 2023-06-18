"""Normalization functions"""

import numpy as np
from numpynn import layers, inits, utils


class Normalization(layers.ParamLayer):
    """Normalization layer base class"""


class Layernorm(Normalization):
    """Implements layer normalization."""

    def __init__(self, eps: float = 1e-7):
        super().__init__(None, None, None, True)
        self.eps = eps
        self.__var_inv = None
        self.__xhat = None

    def compile(self, i, prev_layer, succ_layer) -> None:
        super().compile(i, prev_layer, succ_layer)
        gamma = inits.ones((1, prev_layer.y.shape[1]))
        self.g = self.dg = self.g_delta = self.g_m = self.g_v = utils.expand_dims(gamma, prev_layer.y.ndim)
        self.b = self.db = self.b_delta = self.b_m = self.b_v = inits.zeros_like(self.g)
        self.forward()

    def forward(self):
        super().forward()
        std_axis = tuple(i + 1 for i in range(self.x.ndim - 1))
        mean = np.mean(self.x, axis=std_axis, keepdims=True)
        var = np.var(self.x, axis=std_axis, keepdims=True, ddof=1)
        self.__var_inv = (var + self.eps)**-0.5
        self.__xhat = (self.x - mean) * self.__var_inv
        self.y = self.g * self.__xhat + self.b

    def backward(self):
        super().backward()
        sum_axis_bg = (0,) + tuple(i + 2 for i in range(self.x.ndim - 2))
        # gamma gradients
        self.dg = np.sum(self.__xhat * self.dy, axis=sum_axis_bg, keepdims=True)
        # beta gradients
        self.db =  np.sum(self.dy, axis=sum_axis_bg, keepdims=True)
        # input gradients
        n = self.x[0].size
        sum_axis_x = tuple(i + 1 for i in range(self.x.ndim - 1))
        temp1 = n * self.dy
        temp2 = np.sum(self.dy, sum_axis_x, keepdims=True)
        temp3 = n / (n - 1) * self.__xhat * np.sum(self.dy * self.__xhat, sum_axis_x, keepdims=True)
        self.dx = self.g * self.__var_inv / n * (temp1 - temp2 - temp3)
