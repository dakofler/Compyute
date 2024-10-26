"""Neural network shape changing functions."""

from ...tensor_ops.creation_ops import zeros
from ...tensors import ShapeLike, Tensor
from .functions import Function, FunctionCache


class FlattenFn(Function):
    """Flattens tensors not including the batch dimension."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        cache.push(x.shape)
        return x.view((x.shape[0], -1))

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        (x_shape,) = cache.pop()
        return dy.view(x_shape)


class ReshapeFn(Function):
    """Reshapes tensors."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, shape: ShapeLike) -> Tensor:
        cache.push(x.shape)
        return x.view((x.shape[0],) + shape)

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        (x_shape,) = cache.pop()
        return dy.view(x_shape)


class SliceFn(Function):
    """Slices tensors."""

    @staticmethod
    def forward(
        cache: FunctionCache, x: Tensor, slice: tuple[slice | int, ...]
    ) -> Tensor:
        cache.push(x.shape, slice)
        return x[slice]

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x_shape, slice = cache.pop()
        dx = zeros(x_shape, device=dy.device, dtype=dy.dtype)
        dx[slice] = dy
        return dx
