"""Neural network reshaping functions."""

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


class FReshape(Function):
    """Reshapes tensors."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, shape: ShapeLike) -> Tensor:
        cache.push(x.shape)
        return x.view((x.shape[0],) + shape)

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        (x_shape,) = cache.pop()
        return dy.view(x_shape)
