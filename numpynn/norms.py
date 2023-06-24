"""Normalization functions"""

import numpy as np
from numpynn import layers
from numpynn.tensor import ones, match_dims, zeros_like


class Normalization(layers.ParamLayer):
    """Normalization layer base class"""


class Layernorm(Normalization):
    """Implements layer normalization.

    ### Parameters
        eps: `float`, optional
            Constant for numerical stability.
    """
    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__(None, None, None, True)
        self.eps = eps
        self._var_inv = None
        self._xhat = None
        self.g = None

    def compile(self, i: int, prev_layer: layers.Layer, succ_layer: layers.Layer) -> None:
        super().compile(i, prev_layer, succ_layer)
        gamma = ones((1, self.prev_layer.y.shape[1]))
        self.g = match_dims(gamma, self.prev_layer.y.ndim)
        self.b = zeros_like(self.g.data)
        self.params = [self.g, self.b]
        self.forward()

    def forward(self) -> None:
        super().forward()
        std_axis = tuple(i + 1 for i in range(self.x.ndim - 1))
        mean = np.mean(self.x.data, axis=std_axis, keepdims=True)
        var = np.var(self.x.data, axis=std_axis, keepdims=True, ddof=1)
        self._var_inv = (var + self.eps)**-0.5
        self._xhat = (self.x.data - mean) * self._var_inv
        self.y.data = self.g.data * self._xhat + self.b.data

    def backward(self) -> None:
        super().backward()
        sum_axis_bg = (0,) + tuple(i + 2 for i in range(self.x.ndim - 2))
        # gamma gradients
        self.g.grad = np.sum(self._xhat * self.y.grad, axis=sum_axis_bg, keepdims=True)
        # beta gradients
        self.b.grad =  np.sum(self.y.grad, axis=sum_axis_bg, keepdims=True)
        # input gradients
        n = self.x.data[0].size
        sum_axis_x = tuple(i + 1 for i in range(self.x.ndim - 1))
        temp1 = n * self.y.grad
        temp2 = np.sum(self.y.grad, sum_axis_x, keepdims=True)
        temp3 = n / (n - 1) * self._xhat * np.sum(self.y.grad * self._xhat, sum_axis_x, keepdims=True)
        self.x.grad = self.g.data * self._var_inv / n * (temp1 - temp2 - temp3)
