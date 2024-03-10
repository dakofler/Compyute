"""Normalization layers module"""

from compyute.functional import ones, prod, zeros
from compyute.nn.module import Module
from compyute.nn.parameter import Parameter
from compyute.tensor import Tensor, ArrayLike, ShapeLike


__all__ = ["Batchnorm1d", "Batchnorm2d", "Layernorm"]


class Batchnorm1d(Module):
    """Batch Normalization."""

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        m: float = 0.1,
        dtype: str = "float32",
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
            Number of channels of the layer.
        eps : float, optional
            Constant for numerical stability, by default 1e-5.
        m : float, optional
            Momentum used for running mean and variance computation, by default 0.1.
        dtype: str, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.m = m
        self.dtype = dtype

        # parameters
        self.w = Parameter(ones((channels,)), dtype=dtype, label="w")
        self.b = Parameter(zeros((channels,)), dtype=dtype, label="b")

        # buffers
        self.rmean = zeros((channels,), dtype=dtype)
        self.rvar = ones((channels,), dtype=dtype)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        channels = self.channels
        eps = self.eps
        m = self.m
        dtype = self.dtype
        return f"{name}({channels=}, {eps=}, {m=}, {dtype=})"

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [2, 3])
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

            def backward(dy: ArrayLike) -> ArrayLike:
                dy = dy.astype(self.dtype)
                self.set_dy(dy)

                # input grads
                n = float(prod(x.shape) / x.shape[1])
                dx = (
                    weights.data
                    * var_h.data
                    / n
                    * (
                        n * dy
                        - dy.sum(axis=axis, keepdims=True)
                        - x_h.data * (dy * x_h.data).sum(axis=axis, keepdims=True)
                    )
                )

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
        dtype: str = "float32",
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
            Number of channels of the layer.
        eps : float, optional
            Constant for numerical stability, by default 1e-5.
        m : float, optional
            Momentum used for running mean and variance computation, by default 0.1.
        dtype: str, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.m = m
        self.dtype = dtype

        # parameters
        self.w = Parameter(ones((channels,)), dtype=dtype, label="w")
        self.b = Parameter(zeros((channels,)), dtype=dtype, label="b")

        # buffers
        self.rmean = zeros((channels,), dtype=dtype)
        self.rvar = ones((channels,), dtype=dtype)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        channels = self.channels
        eps = self.eps
        m = self.m
        dtype = self.dtype
        return f"{name}({channels=}, {eps=}, {m=}, {dtype=})"

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [4])
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

            def backward(dy: ArrayLike) -> ArrayLike:
                dy = dy.astype(self.dtype)
                self.set_dy(dy)

                # input grads
                n = float(prod(x.shape) / x.shape[1])
                dx = (
                    weights.data
                    * var_h.data
                    / n
                    * (
                        n * dy
                        - dy.sum(axis=axis, keepdims=True)
                        - x_h.data * (dy * x_h.data).sum(axis=axis, keepdims=True)
                    )
                )

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

        # necessary, because Module.to_device only moves parameters
        self.rmean.to_device(device)
        self.rvar.to_device(device)

        super().to_device(device)


class Layernorm(Module):
    """Normalizes values per sample."""

    def __init__(
        self, normalized_shape: ShapeLike, eps: float = 1e-5, dtype: str = "float32"
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
        dtype: str, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.dtype = dtype

        # parameters
        self.w = Parameter(ones(normalized_shape), dtype=dtype, label="w")
        self.b = Parameter(zeros(normalized_shape), dtype=dtype, label="b")

    def __repr__(self) -> str:
        name = self.__class__.__name__
        normalized_shape = self.normalized_shape
        eps = self.eps
        dtype = self.dtype
        return f"{name}({normalized_shape=}, {eps=}, {dtype=})"

    def forward(self, x: Tensor) -> Tensor:
        x = x.astype(self.dtype)

        axes = tuple([i for i in range(1, x.ndim)])
        var_h = (x.var(axis=axes, keepdims=True) + self.eps) ** -0.5
        x_h = (x - x.mean(axis=axes, keepdims=True)) * var_h
        y = self.w * x_h + self.b

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                dy = dy.astype(self.dtype)
                self.set_dy(dy)

                # input grads
                n = prod(x.shape[1:])
                dx = (
                    self.w.data
                    * var_h.data
                    / n
                    * (
                        n * dy
                        - dy.sum(axes, keepdims=True)
                        - x_h.data * (dy * x_h.data).sum(axes, keepdims=True)
                    )
                )

                # gamma grads
                self.w.grad = (x_h.data * dy).sum(axis=0)

                # beta grads
                self.b.grad = dy.sum(axis=0)

                return dx

            self.backward = backward

        self.set_y(y)
        return y
