"""Normalization layers module"""

import numpy as np

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ShapeLike, NumpyArray, AxisLike
from walnut.nn.layers.parameter import Parameter


__all__ = ["Batchnorm", "Layernorm"]


class Batchnorm(Parameter):
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
            Shape of a sample. Required if the layer is used as input, by default None.
        """
        super().__init__(input_shape=input_shape)
        self.eps = eps
        self.momentum = momentum
        self.rmean: Tensor | None = None
        self.rvar: Tensor | None = None
        self._ax: AxisLike | None = None

    def compile(self) -> None:
        super().compile()
        self.w = tu.ones((self.x.shape[1],))
        self.b = tu.zeros_like(self.w)
        self.parameters = [self.w, self.b]
        self._ax = (0,) + tuple(i + 2 for i in range(self.x.ndim - 2))
        self.rmean = tu.zeros(self.x.mean(axis=self._ax, keepdims=True).shape)
        self.rvar = tu.ones_like(self.rmean)

    def __call__(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.mean(axis=self._ax, keepdims=True)
            var = x.var(axis=self._ax, keepdims=True)
            var_h = (var + self.eps) ** -0.5
            x_h = (x - mean) * var_h

            # keep running stats
            self.rmean = self.rmean * (1.0 - self.momentum) + mean * self.momentum
            # for some reason, torch uses ddof=1 here
            var_temp = x.var(axis=self._ax, keepdims=True, ddof=1)
            self.rvar = self.rvar * (1.0 - self.momentum) + var_temp * self.momentum
        else:
            var_h = (self.rvar + self.eps) ** -0.5
            x_h = (x - self.rmean) * var_h

        weights = tu.match_dims(self.w, dims=x.ndim - 1)
        bias = tu.match_dims(self.b, dims=x.ndim - 1)
        y = weights * x_h + bias

        if self.training:

            def backward(y_grad: NumpyArray) -> NumpyArray:
                # input grads
                n = np.prod(x.shape) / x.shape[1]
                tmp1 = n * y_grad
                tmp2 = np.sum(y_grad, axis=self._ax, keepdims=True)
                tmp3 = np.sum(y_grad * x_h.data, axis=self._ax, keepdims=True)
                weights = tu.match_dims(self.w, dims=x.ndim - 1).data
                x_grad = weights * var_h.data / n * (tmp1 - tmp2 - x_h.data * tmp3)

                # gamma grads
                self.w.grad = np.sum(x_h.data * y_grad, axis=self._ax)

                # beta grads
                self.b.grad = np.sum(y_grad, axis=self._ax)

                self.set_y_grad(y_grad)
                self.set_x_grad(x_grad)
                return x_grad

            self.backward = backward

        self.set_x(x)
        self.set_y(y)
        return y


class Layernorm(Parameter):
    """Normalizes values per sample."""

    def __init__(self, eps: float = 1e-5, input_shape: ShapeLike | None = None) -> None:
        """Implements layer normalization.

        Parameters
        ----------
        eps : float, optional
            Constant for numerical stability, by default 1e-5.
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the layer is used as input, by default None.
        """
        super().__init__(input_shape=input_shape)
        self.eps = eps
        self._ax: AxisLike | None = None

    def compile(self) -> None:
        super().compile()
        self.w = tu.ones(self.x.shape[1:])
        self.b = tu.zeros_like(self.w)
        self.parameters = [self.w, self.b]
        self._ax = tuple(i + 1 for i in range(self.x.ndim - 1))

    def __call__(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=self._ax, keepdims=True)
        var = x.var(axis=self._ax, keepdims=True)
        var_h = (var + self.eps) ** -0.5
        x_h = (x - mean) * var_h
        y = self.w * x_h + self.b

        if self.training:

            def backward(y_grad: NumpyArray) -> NumpyArray:
                # input grads
                n = x.data[0].size
                tmp1 = n * y_grad
                tmp2 = np.sum(y_grad, self._ax, keepdims=True)
                tmp3 = np.sum(y_grad * x_h.data, self._ax, keepdims=True)
                x_grad = self.w.data * var_h.data / n * (tmp1 - tmp2 - x_h.data * tmp3)

                # gamma grads
                self.w.grad = np.sum(x_h.data * y_grad, axis=0)

                # beta grads
                self.b.grad = np.sum(y_grad, axis=0)

                self.set_y_grad(y_grad)
                self.set_x_grad(x_grad)
                return x_grad

            self.backward = backward

        self.set_x(x)
        self.set_y(y)
        return y


NORMALIZATIONS = {"layer": Layernorm, "batch": Batchnorm}
