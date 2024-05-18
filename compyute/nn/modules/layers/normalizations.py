"""Normalization layers module"""

from typing import Optional

from ....tensor_functions import ones, tensorprod, zeros
from ....tensors import Tensor
from ....types import DtypeLike, ShapeLike
from ...parameter import Parameter
from ..module import Module

__all__ = ["Batchnorm1d", "Batchnorm2d", "Layernorm"]


class Batchnorm1d(Module):
    """Batch Normalization."""

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        m: float = 0.1,
        dtype: DtypeLike = "float32",
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
        self.rmean = zeros((channels,), dtype)
        self.rvar = ones((channels,), dtype)

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [2, 3])
        x = x.astype(self.dtype)

        dim2 = x.ndim == 2
        axis = 0 if dim2 else (0, 2)
        if self.training:
            mean = x.mean(axis=axis, keepdims=True)
            var = x.var(axis=axis, keepdims=True)
            var_h = (var + self.eps) ** -0.5
            x_h = (x - mean) * var_h

            # keep running stats
            self.rmean = self.rmean * (1 - self.m) + mean.squeeze() * self.m
            var = x.var(axis=axis, keepdims=True, ddof=1)
            self.rvar = self.rvar * (1 - self.m) + var.squeeze() * self.m
        else:
            rvar = self.rvar if dim2 else self.rvar.reshape((*self.rvar.shape, 1))
            rmean = self.rmean if dim2 else self.rmean.reshape((*self.rmean.shape, 1))
            var_h = (rvar + self.eps) ** -0.5
            x_h = (x - rmean) * var_h

        weights = self.w if dim2 else self.w.reshape((*self.w.shape, 1))
        biases = self.b if dim2 else self.b.reshape((*self.b.shape, 1))
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
                        - dy.sum(axis=axis, keepdims=True)
                        - x_h * (dy * x_h).sum(axis=axis, keepdims=True)
                    )
                )

                # gamma grads
                if self.w.requires_grad:
                    self.w.grad = (x_h * dy).sum(axis=axis)

                # beta grads
                if self.b.requires_grad:
                    self.b.grad = dy.sum(axis=axis)

                return dx

            self._backward = _backward

        return y

    def to_device(self, device: str) -> None:
        """Moves the tensor to a specified device.

        Parameters
        ----------
        device : str
            Device to move the tensor to. Valid options are "cpu" and "cuda".
        """
        self.rmean.to_device(device)
        self.rvar.to_device(device)
        super().to_device(device)


class Batchnorm2d(Module):
    """Batch Normalization."""

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        m: float = 0.1,
        dtype: DtypeLike = "float32",
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
        self.rmean = zeros((channels,), dtype)
        self.rvar = ones((channels,), dtype)

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [4])
        x = x.astype(self.dtype)

        axis = (0, 2, 3)
        if self.training:
            mean = x.mean(axis=axis, keepdims=True)
            var = x.var(axis=axis, keepdims=True)
            var_h = (var + self.eps) ** -0.5
            x_h = (x - mean) * var_h

            # keep running stats
            self.rmean = self.rmean * (1 - self.m) + mean.squeeze() * self.m
            var = x.var(axis=axis, keepdims=True, ddof=1)
            self.rvar = self.rvar * (1 - self.m) + var.squeeze() * self.m
        else:
            rvar = self.rvar.reshape((*self.rvar.shape, 1, 1))
            rmean = self.rmean.reshape((*self.rmean.shape, 1, 1))
            var_h = (rvar + self.eps) ** -0.5
            x_h = (x - rmean) * var_h

        weights = self.w.reshape((*self.w.shape, 1, 1))
        biases = self.b.reshape((*self.b.shape, 1, 1))
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
                        - dy.sum(axis=axis, keepdims=True)
                        - x_h * (dy * x_h).sum(axis=axis, keepdims=True)
                    )
                )

                # gamma grads
                if self.w.requires_grad:
                    self.w.grad = (x_h * dy).sum(axis=axis)

                # beta grads
                if self.b.requires_grad:
                    self.b.grad = dy.sum(axis=axis)

                return dx

            self._backward = _backward

        return y

    def to_device(self, device: str) -> None:
        """Moves the tensor to a specified device.

        Parameters
        ----------
        device : str
            Device to move the tensor to. Valid options are "cpu" and "cuda".
        """

        # necessary, because Module.to_device only moves parameters
        super().to_device(device)
        self.rmean.to_device(device)
        self.rvar.to_device(device)


class Layernorm(Module):
    """Normalizes values per sample."""

    def __init__(
        self,
        normalized_shape: ShapeLike,
        eps: float = 1e-5,
        dtype: DtypeLike = "float32",
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
        var_h = (x.var(axis=axes, keepdims=True) + self.eps) ** -0.5
        x_h = (x - x.mean(axis=axes, keepdims=True)) * var_h
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
                        - dy.sum(axes, keepdims=True)
                        - x_h * (dy * x_h).sum(axes, keepdims=True)
                    )
                )

                # gamma grads
                if self.w.requires_grad:
                    self.w.grad = (x_h * dy).sum(axis=0)

                # beta grads
                if self.b.requires_grad:
                    self.b.grad = dy.sum(axis=0)

                return dx

            self._backward = _backward

        return y
