"""Normalization layers module"""

import compyute.tensor_functions as tf
from compyute.tensor import Tensor, ArrayLike, ShapeLike
from compyute.nn.module import Module
from compyute.nn.parameter import Parameter


__all__ = ["Batchnorm", "Layernorm"]


class Batchnorm(Module):
    """Batch Normalization."""

    def __init__(
        self,
        in_channels: int,
        eps: float = 1e-5,
        m: float = 0.1,
        dtype: str = "float32",
    ) -> None:
        """Implements Batch Normalization.

        Parameters
        ----------
        in_channels : int
            Number of input channels of the layer.
        eps : float, optional
            Constant for numerical stability, by default 1e-5.
        m : float, optional
            Momentum used for running mean and variance computation, by default 0.1.
        dtype: str, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.m = m
        self.dtype = dtype

        # parameters
        self.w = Parameter(tf.ones((in_channels,)), dtype=dtype, label="w")
        self.b = Parameter(tf.zeros((in_channels,)), dtype=dtype, label="b")

        # buffers
        self.rmean = tf.zeros((in_channels,), dtype=dtype, device=self.device)
        self.rvar = tf.ones((in_channels,), dtype=dtype, device=self.device)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        in_channels = self.in_channels
        eps = self.eps
        m = self.m
        dtype = self.dtype
        return f"{name}({in_channels=}, {eps=}, {m=}, {dtype=})"

    def __call__(self, x: Tensor) -> Tensor:
        x = x.astype(self.dtype)

        axis = (0,) + tuple(tf.arange(x.ndim).data[2:])
        if self.training:
            mean = x.mean(axis=axis, keepdims=True)
            var = x.var(axis=axis, keepdims=True)
            var_h = (var + self.eps) ** -0.5
            x_h = (x - mean) * var_h

            # keep running stats
            self.rmean = self.rmean * (1.0 - self.m) + mean.squeeze() * self.m
            _var = x.var(axis=axis, keepdims=True, ddof=1)
            self.rvar = self.rvar * (1.0 - self.m) + _var.squeeze() * self.m
        else:
            _rvar = tf.match_dims(self.rvar, x.ndim - 1)
            _rmean = tf.match_dims(self.rmean, x.ndim - 1)
            var_h = (_rvar + self.eps) ** -0.5
            x_h = (x - _rmean) * var_h

        weights = tf.match_dims(self.w, dims=x.ndim - 1)
        bias = tf.match_dims(self.b, dims=x.ndim - 1)
        y = weights * x_h + bias

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                dy = dy.astype(self.dtype)
                self.set_dy(dy)

                # input grads
                n = float(tf.prod(x.shape) / x.shape[1])
                tmp1 = n * dy
                tmp2 = dy.sum(axis=axis, keepdims=True)
                tmp3 = (dy * x_h.data).sum(axis=axis, keepdims=True)
                dx = weights.data * var_h.data / n * (tmp1 - tmp2 - x_h.data * tmp3)

                # gamma grads
                self.w.grad = (x_h.data * dy).sum(axis=axis)

                # beta grads
                self.b.grad = dy.sum(axis=axis)

                return dx

            self.backward = backward

        self.set_y(y)
        return y

    def to_device(self, device: str) -> None:
        """Moves the tensor to a specified device.

        Parameters
        ----------
        device : str
            Device to move the tensor to. Valid options are "cpu" and "cuda".
        """
        super().to_device(device)
        self.rmean.to_device(device)
        self.rvar.to_device(device)


class Layernorm(Module):
    """Normalizes values per sample."""

    def __init__(
        self, normalized_shape: ShapeLike, eps: float = 1e-5, dtype: str = "float32"
    ) -> None:
        """Implements layer normalization.

        Parameters
        ----------
        normalized_shape : ShapeLike
            Shape of the normalized tensor ignoring the batch dimension.
        eps : float, optional
            Constant for numerical stability, by default 1e-5.
        dtype: str, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.dtype = dtype

        # parameters
        self.w = Parameter(tf.ones(normalized_shape), dtype=dtype, label="w")
        self.b = Parameter(tf.zeros(normalized_shape), dtype=dtype, label="b")

    def __repr__(self) -> str:
        name = self.__class__.__name__
        normalized_shape = self.normalized_shape
        eps = self.eps
        dtype = self.dtype
        return f"{name}({normalized_shape=}, {eps=}, {dtype=})"

    def __call__(self, x: Tensor) -> Tensor:
        x = x.astype(self.dtype)

        axis = tuple(tf.arange(x.ndim).data[1:])
        mean = x.mean(axis=axis, keepdims=True)
        var = x.var(axis=axis, keepdims=True)
        var_h = (var + self.eps) ** -0.5
        x_h = (x - mean) * var_h
        y = self.w * x_h + self.b

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                dy = dy.astype(self.dtype)
                self.set_dy(dy)

                # input grads
                n = x.data[0].size
                tmp1 = n * dy
                tmp2 = dy.sum(axis, keepdims=True)
                tmp3 = (dy * x_h.data).sum(axis, keepdims=True)
                dx = self.w.data * var_h.data / n * (tmp1 - tmp2 - x_h.data * tmp3)

                # gamma grads
                self.w.grad = (x_h.data * dy).sum(axis=0)

                # beta grads
                self.b.grad = dy.sum(axis=0)

                return dx

            self.backward = backward

        self.set_y(y)
        return y
