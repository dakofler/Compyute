"""Normalization layers module"""

from typing import Optional

from ....base_tensor import Tensor
from ....tensor_functions.computing import tensorprod
from ....tensor_functions.creating import ones, zeros
from ....tensor_functions.reshaping import reshape, squeeze
from ....tensor_functions.transforming import mean as _mean
from ....tensor_functions.transforming import sum as _sum
from ....tensor_functions.transforming import var as _var
from ....types import _DtypeLike, _ShapeLike
from ...parameter import Buffer, Parameter
from ..module import Module

__all__ = ["Batchnorm1d", "Batchnorm2d", "Layernorm"]


class Batchnorm1d(Module):
    """Batch Normalization."""

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        m: float = 0.1,
        dtype: _DtypeLike = "float32",
        label: Optional[str] = None,
    ) -> None:
        """Implements Batch Normalization.
        Input: (B, C, T) or (B, C)
            B ... batch, C ... channels, T ... time
        Output: (B, C, T) or (B, C)
            B ... batch, C ... channels, T ... time
        Normalizes over the C dimension.

        Parameters
        ----------
        channels : int
            Number of channels.
        eps : float, optional
            Constant for numerical stability, by default 1e-5.
        m : float, optional
            Momentum used for running mean and variance computation, by default 0.1.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        label: str, optional
            Module label.
        """
        super().__init__(label)
        self.channels = channels
        self.eps = eps
        self.m = m
        self.dtype = dtype

        # parameters
        self.w = Parameter(ones((channels,), dtype), label="w")
        self.b = Parameter(zeros((channels,), dtype), label="b")

        # buffers
        self.rmean = Buffer(zeros((channels,), dtype), label="rmean")
        self.rvar = Buffer(ones((channels,), dtype), label="rvar")

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [2, 3])
        x = x.astype(self.dtype)

        dim2 = x.ndim == 2
        axis = 0 if dim2 else (0, 2)
        if self.training:
            mean = _mean(x, axis=axis, keepdims=True)
            var = _var(x, axis=axis, keepdims=True)
            var_h = (var + self.eps) ** -0.5
            x_h = (x - mean) * var_h

            # keep running stats
            self.rmean = self.rmean * (1 - self.m) + squeeze(mean) * self.m
            var = _var(x, axis=axis, keepdims=True, ddof=1)
            self.rvar = self.rvar * (1 - self.m) + squeeze(var) * self.m
        else:
            rvar = self.rvar if dim2 else reshape(self.rvar, shape=(*self.rvar.shape, 1))
            rmean = self.rmean if dim2 else reshape(self.rmean, shape=(*self.rmean.shape, 1))
            var_h = (rvar + self.eps) ** -0.5
            x_h = (x - rmean) * var_h

        weights = self.w if dim2 else reshape(self.w, shape=(*self.w.shape, 1))
        biases = self.b if dim2 else reshape(self.b, shape=(*self.b.shape, 1))
        y = weights * x_h + biases

        if self.training:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.astype(self.dtype)

                # input grads
                n = tensorprod(x.shape) / x.shape[1]
                dx = (
                    weights
                    * var_h
                    / n
                    * (
                        n * dy
                        - _sum(dy, axis=axis, keepdims=True)
                        - x_h * _sum(dy * x_h, axis=axis, keepdims=True)
                    )
                )

                # gamma grads
                if self.w.requires_grad:
                    self.w.grad = _sum(x_h * dy, axis=axis)

                # beta grads
                if self.b.requires_grad:
                    self.b.grad = _sum(dy, axis=axis)

                return dx

            self._backward = _backward

        return y


class Batchnorm2d(Module):
    """Batch Normalization."""

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        m: float = 0.1,
        dtype: _DtypeLike = "float32",
        label: Optional[str] = None,
    ) -> None:
        """Implements Batch Normalization.
        Input: (B, C, Y, X)
            B ... batch, C ... channels, Y ... height, X ... width
        Output: (B, C, Y, X)
            B ... batch, C ... channels, Y ... height, X ... width
        Normalizes over the C dimension.

        Parameters
        ----------
        channels : int
            Number of channels.
        eps : float, optional
            Constant for numerical stability, by default 1e-5.
        m : float, optional
            Momentum used for running mean and variance computation, by default 0.1.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        label: str, optional
            Module label.
        """
        super().__init__(label)
        self.channels = channels
        self.eps = eps
        self.m = m
        self.dtype = dtype

        # parameters
        self.w = Parameter(ones((channels,), dtype), label="w")
        self.b = Parameter(zeros((channels,), dtype), label="b")

        # buffers
        self.rmean = Buffer(zeros((channels,), dtype), label="rmean")
        self.rvar = Buffer(ones((channels,), dtype), label="rvar")

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [4])
        x = x.astype(self.dtype)

        axis = (0, 2, 3)
        if self.training:
            mean = _mean(x, axis=axis, keepdims=True)
            var = _var(x, axis=axis, keepdims=True)
            var_h = (var + self.eps) ** -0.5
            x_h = (x - mean) * var_h

            # keep running stats
            self.rmean = self.rmean * (1 - self.m) + squeeze(mean) * self.m
            var = _var(x, axis=axis, keepdims=True, ddof=1)
            self.rvar = self.rvar * (1 - self.m) + squeeze(var) * self.m
        else:
            rvar = reshape(self.rvar, shape=(*self.rvar.shape, 1, 1))
            rmean = reshape(self.rmean, shape=(*self.rmean.shape, 1, 1))
            var_h = (rvar + self.eps) ** -0.5
            x_h = (x - rmean) * var_h

        weights = reshape(self.w, shape=(*self.w.shape, 1, 1))
        biases = reshape(self.b, shape=(*self.b.shape, 1, 1))
        y = weights * x_h + biases

        if self.training:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.astype(self.dtype)

                # input grads
                n = tensorprod(x.shape) / x.shape[1]
                dx = (
                    weights
                    * var_h
                    / n
                    * (
                        n * dy
                        - _sum(dy, axis=axis, keepdims=True)
                        - x_h * _sum(dy * x_h, axis=axis, keepdims=True)
                    )
                )

                # gamma grads
                if self.w.requires_grad:
                    self.w.grad += _sum(x_h * dy, axis=axis)

                # beta grads
                if self.b.requires_grad:
                    self.b.grad += _sum(dy, axis=axis)

                return dx

            self._backward = _backward

        return y


class Layernorm(Module):
    """Normalizes values per sample."""

    def __init__(
        self,
        normalized_shape: _ShapeLike,
        eps: float = 1e-5,
        dtype: _DtypeLike = "float32",
        label: Optional[str] = None,
    ) -> None:
        """Implements layer normalization.
        Input: (B, ...)
            B ... batch
        Output: (B, ...)
            B ... batch
        Normalizes over all trailing dimensions.

        Parameters
        ----------
        normalized_shape : ShapeLike
            Shape of the normalized tensor ignoring the batch dimension.
        eps : float, optional
            Constant for numerical stability, by default 1e-5.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        label: str, optional
            Module label.
        """
        super().__init__(label)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.dtype = dtype

        # parameters
        self.w = Parameter(ones(normalized_shape, dtype), label="w")
        self.b = Parameter(zeros(normalized_shape, dtype), label="b")

    def forward(self, x: Tensor) -> Tensor:
        x = x.astype(self.dtype)

        axes = tuple([i for i in range(1, x.ndim)])
        var_h = (_var(x, axis=axes, keepdims=True) + self.eps) ** -0.5
        x_h = (x - _mean(x, axis=axes, keepdims=True)) * var_h
        y = self.w * x_h + self.b

        if self.training:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.astype(self.dtype)

                # input grads
                n = tensorprod(x.shape[1:])
                dx = (
                    self.w
                    * var_h
                    / n
                    * (
                        n * dy
                        - _sum(dy, axes, keepdims=True)
                        - x_h * _sum(dy * x_h, axes, keepdims=True)
                    )
                )

                # gamma grads
                if self.w.requires_grad:
                    self.w.grad += _sum(x_h * dy, axis=0)

                # beta grads
                if self.b.requires_grad:
                    self.b.grad += _sum(dy, axis=0)

                return dx

            self._backward = _backward

        return y
