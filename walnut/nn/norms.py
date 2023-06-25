"""Normalization functions"""

import numpy as np

from walnut import tensor
from walnut.tensor import Tensor
from walnut.nn import layers


class Normalization(layers.ParamLayer):
    """Normalization layer base class"""


class Layernorm(Normalization):
    """Implements layer normalization.

    ### Parameters
        eps: `float`, optional
            Constant for numerical stability.
    """

    __slots__ = 'eps', '_var_inv', '_xhat', 'g'

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__(use_bias=True)
        self.eps = eps
        self._var_inv: np.ndarray = None
        self._xhat: np.ndarray = None
        self.g: Tensor = None

    def compile(self, prev_layer: layers.Layer, succ_layer: layers.Layer) -> None:
        super().compile(prev_layer, succ_layer)
        gamma = tensor.ones((1, self.prev_layer.y.shape[1]))
        self.g = tensor.match_dims(gamma, self.prev_layer.y.ndim)
        self.b = tensor.zeros_like(self.g.data)
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
        axis_bg = (0,) + tuple(i + 2 for i in range(self.x.ndim - 2)) # axis tuple for summation
        self.g.grad = np.sum(self._xhat * self.y.grad, axis=axis_bg, keepdims=True) # gamma grads
        self.b.grad =  np.sum(self.y.grad, axis=axis_bg, keepdims=True) # beta grads

        # input grads
        n = self.x.data[0].size
        axis_x = tuple(i + 1 for i in range(self.x.ndim - 1)) # axis tuple for summation
        temp1 = n * self.y.grad
        temp2 = np.sum(self.y.grad, axis_x, keepdims=True)
        temp3 = np.sum(self.y.grad * self._xhat, axis_x, keepdims=True)
        temp4 = n / (n - 1) * self._xhat * temp3
        self.x.grad = self.g.data * self._var_inv / n * (temp1 - temp2 - temp4)
