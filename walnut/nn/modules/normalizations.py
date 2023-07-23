"""Normalization modules module"""

from dataclasses import dataclass
import numpy as np

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ShapeLike, NumpyArray, AxisLike
from walnut.nn.optimizers import Optimizer
from walnut.nn.modules.parameter import ParamModule


__all__ = ["Batchnorm", "Layernorm"]


@dataclass(init=False, repr=False)
class Batchnorm(ParamModule):
    """Batch Normalization."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        input_shape: ShapeLike | None = None,
    ) -> None:
        """Implements Batch Normalization.

        Parameters
        ----------
        eps : float, optional
            Constant for numerical stability, by default 1e-5.
        momentum : float, optional
            Momentum used for running mean and variance computation, by default 0.1.
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the module is used as input, by default None.
        """
        super().__init__(input_shape=input_shape)
        self.eps = eps
        self.momentum = momentum
        self._vari: NumpyArray | None = None
        self._xh: NumpyArray | None = None
        self._rmean: NumpyArray | None = None
        self._rvar: NumpyArray | None = None
        self._ax: AxisLike | None = None

    def compile(self, optimizer: Optimizer | None = None) -> None:
        super().compile(optimizer)
        self.w = tu.ones((self.x.shape[1],))  # gain
        self.parameters.append(self.w)
        self.b = tu.zeros_like(self.w)
        self.parameters.append(self.b)
        self._ax = (0,) + tuple(i + 2 for i in range(self.x.ndim - 2))
        self._rmean = tu.ones(self.x.shape[1:]).data
        self._rvar = tu.ones(self.x.shape[1:]).data

    def __call__(self, x: Tensor) -> Tensor:
        super().__call__(x)
        if self.training:
            mean = np.mean(self.x.data, axis=self._ax, keepdims=True)
            var = np.var(self.x.data, axis=self._ax, keepdims=True)
            # compute a running mean
            if self._rmean is None or self._rmean.mean() == 1:
                self._rmean = mean
                self._rvar = var
            else:
                self._rmean = (1.0 - self.momentum) * self._rmean + self.momentum * mean
                self._rvar = (1.0 - self.momentum) * self._rvar + self.momentum * var

        self._vari = (self._rvar + self.eps) ** -0.5
        self._xh = (self.x.data - self._rmean) * self._vari

        weights = tu.match_dims(self.w, dims=self.x.ndim - 1).data
        bias = tu.match_dims(self.b, dims=self.x.ndim - 1).data
        self.y.data = weights * self._xh + bias
        return self.y

    def backward(self, y_grad: NumpyArray) -> NumpyArray:
        super().backward(y_grad)
        n = np.prod(self.x.shape) / self.x.shape[1]

        # gamma grads
        self.w.grad = np.sum(self._xh * self.y.grad, axis=self._ax)

        # beta grads
        self.b.grad = np.sum(self.y.grad, axis=self._ax)

        # input grads
        temp1 = n * self.y.grad
        temp2 = np.sum(self.y.grad, axis=self._ax, keepdims=True)
        temp3 = np.sum(self.y.grad * self._xh, axis=self._ax, keepdims=True)
        weights = tu.match_dims(self.w, dims=self.x.ndim - 1).data
        self.x.grad = weights * self._vari / n * (temp1 - temp2 - self._xh * temp3)

        if self.optimizer:
            self.optimize()
        return self.x.grad


@dataclass(init=False, repr=False)
class Layernorm(ParamModule):
    """Normalizes values per sample."""

    def __init__(self, eps: float = 1e-5, input_shape: ShapeLike | None = None) -> None:
        """Implements module normalization.

        Parameters
        ----------
        eps : float, optional
            Constant for numerical stability, by default 1e-5.
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the module is used as input, by default None.
        """
        super().__init__(input_shape=input_shape)
        self.eps = eps
        self._vari: NumpyArray | None = None
        self._xh: NumpyArray | None = None
        self._ax: AxisLike | None = None

    def compile(self, optimizer: Optimizer | None = None) -> None:
        super().compile(optimizer)
        self.w = tu.ones(self.x.shape[1:])  # gain
        self.parameters.append(self.w)
        self.b = tu.zeros_like(self.w)
        self.parameters.append(self.b)
        self._ax = tuple(i + 1 for i in range(self.x.ndim - 1))

    def __call__(self, x: Tensor) -> Tensor:
        super().__call__(x)
        mean = np.mean(self.x.data, axis=self._ax, keepdims=True)
        var = np.var(self.x.data, axis=self._ax, keepdims=True)
        self._vari = (var + self.eps) ** -0.5
        self._xh = (self.x.data - mean) * self._vari
        self.y.data = self.w.data * self._xh + self.b.data
        return self.y

    def backward(self, y_grad: NumpyArray) -> NumpyArray:
        super().backward(y_grad)
        # input grads
        n = self.x.data[0].size
        temp1 = n * self.y.grad
        temp2 = np.sum(self.y.grad, self._ax, keepdims=True)
        temp3 = np.sum(self.y.grad * self._xh, self._ax, keepdims=True)
        self.x.grad = self.w.data * self._vari / n * (temp1 - temp2 - self._xh * temp3)
        # gamma grads
        self.w.grad = np.sum(self._xh * self.y.grad, axis=0)

        # beta grads
        self.b.grad = np.sum(self.y.grad, axis=0)

        if self.optimizer:
            self.optimize()
        return self.x.grad


NORMALIZATIONS = {"layer": Layernorm, "batch": Batchnorm}
