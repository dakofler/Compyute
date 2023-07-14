"""Normalization modules module"""

from dataclasses import dataclass
import numpy as np

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ShapeLike, NumpyArray
from walnut.nn.optimizers import Optimizer
from walnut.nn.modules.parameter import ParamModule


@dataclass(init=False, repr=False)
class Layernorm(ParamModule):
    """Normalizes values per sample."""

    def __init__(self, eps: float = 1e-7, input_shape: ShapeLike | None = None) -> None:
        """Implements module normalization.

        Parameters
        ----------
        eps : float, optional
            Constant for numerical stability, by default 1e-7.
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the module is used as input, by default None.
        """
        super().__init__(input_shape=input_shape)
        self.eps = eps
        self._var_inv: np.ndarray | None = None
        self._xhat: np.ndarray | None = None

    def compile(self, optimizer: Optimizer | None = None) -> None:
        super().compile(optimizer)
        self.w = tu.ones(self.x.shape[1:])  # gain
        self.parameters.append(self.w)
        self.b = tu.zeros_like(self.w)
        self.parameters.append(self.b)

    def __call__(self, x: Tensor) -> Tensor:
        super().__call__(x)
        std_axis = tuple(i + 1 for i in range(self.x.ndim - 1))
        mean = np.mean(self.x.data, axis=std_axis, keepdims=True)
        var = np.var(self.x.data, axis=std_axis, keepdims=True, ddof=1)
        self._var_inv = (var + self.eps) ** -0.5
        self._xhat = (self.x.data - mean) * self._var_inv
        self.y.data = self.w.data * self._xhat + self.b.data
        return self.y

    def backward(self, y_grad: NumpyArray) -> NumpyArray:
        super().backward(y_grad)
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
        return self.x.grad


NORMALIZATIONS = {"layer": Layernorm}
