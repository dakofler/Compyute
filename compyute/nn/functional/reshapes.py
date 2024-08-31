"""Neural network reshaping functions."""

from ...tensors import ShapeLike, Tensor
from .functions import Function, FunctionCache


class Fflatten(Function):
    """Flattens tensors not including the batch dimension."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        cache.x_shape = x.shape
        return x.to_shape((x.shape[0], -1))

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x_shape = cache.x_shape
        return dy.to_shape(x_shape)


class FReshape(Function):
    """Reshapes tensors."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, shape: ShapeLike) -> Tensor:
        cache.x_shape = x.shape
        return x.to_shape((x.shape[0],) + shape)

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x_shape = cache.x_shape
        return dy.to_shape(x_shape)
