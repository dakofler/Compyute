"""Neural network reshaping functions."""

from ...tensor_ops.reshaping import reshape
from ...tensors import ShapeLike, Tensor
from .functions import Function, FunctionCache


class Fflatten(Function):
    """Flattens tensors not including the batch dimension."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor) -> Tensor:
        cache.flat_x = x
        return reshape(x, (x.shape[0], -1))

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        return reshape(dy, cache.flat_x.shape)


class FReshape(Function):
    """Reshapes tensors."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, shape: ShapeLike) -> Tensor:
        cache.reshp_x = x
        return reshape(x, (x.shape[0],) + shape)

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        return reshape(dy, cache.reshp_x.shape)
