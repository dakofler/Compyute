"""Normalization layers module"""

from dataclasses import dataclass
import numpy as np

from walnut import tensor
from walnut.nn.optimizers import Optimizer
from walnut.nn.layers.parameter import ParamLayer


@dataclass(init=False, repr=False)
class Layernorm(ParamLayer):
    """Implements layer normalization."""

    def __init__(
        self, eps: float = 1e-7, input_shape: tuple[int, ...] | None = None
    ) -> None:
        """Implements layer normalization.

        Parameters
        ----------
        eps : float, optional
            Constant for numerical stability, by default 1e-7.
        input_shape : tuple[int, ...] | None, optional
            Shape of a sample. Required if the layer is used as input, by default None.
        """
        super().__init__(input_shape=input_shape)
        self.eps = eps
        self._var_inv: np.ndarray | None = None
        self._xhat: np.ndarray | None = None

    def compile(self, optimizer: Optimizer | None = None) -> None:
        super().compile(optimizer)
        self.w = tensor.ones(self.x.shape[1:])  # gain
        self.b = tensor.zeros_like(self.w)
        self.parameters = [self.w, self.b]

    def forward(self, mode: str = "eval") -> None:
        super().forward()
        std_axis = tuple(i + 1 for i in range(self.x.ndim - 1))
        mean = np.mean(self.x.data, axis=std_axis, keepdims=True)
        var = np.var(self.x.data, axis=std_axis, keepdims=True, ddof=1)
        self._var_inv = (var + self.eps) ** -0.5
        self._xhat = (self.x.data - mean) * self._var_inv
        self.y.data = self.w.data * self._xhat + self.b.data

    def backward(self) -> None:
        super().backward()
        n = self.x.data[0].size
        # axis tuple for summation
        axis_x = tuple(i + 1 for i in range(self.x.ndim - 1))
        # gamma grads
        self.w.grad = np.sum(self._xhat * self.y.grad, axis=0)
        # beta grads
        self.b.grad = np.sum(self.y.grad, axis=0)
        # input grads
        temp1 = n * self.y.grad
        temp2 = np.sum(self.y.grad, axis_x, keepdims=True)
        temp3 = np.sum(self.y.grad * self._xhat, axis_x, keepdims=True)
        temp4 = n / (n - 1) * self._xhat * temp3
        self.x.grad = self.w.data * self._var_inv / n * (temp1 - temp2 - temp4)
        if self.optimizer:
            self.optimize()


NORMALIZATIONS = {"layer": Layernorm}
