"""Neural network reshaping modules."""

from typing import Optional

from ...tensors import ShapeLike, Tensor
from ..functional.reshape_funcs import FlattenFn, ReshapeFn
from .module import Module

__all__ = ["Flatten", "Reshape"]


class Flatten(Module):
    """Flattes tensors not including the batch dimension.

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return FlattenFn.forward(self.fcache, x)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return FlattenFn.backward(self.fcache, dy)


class Reshape(Module):
    """Reshapes tensors.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the output tensor.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(self, shape: ShapeLike, label: Optional[str] = None) -> None:
        super().__init__(label)
        self.shape = shape

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return ReshapeFn.forward(self.fcache, x, self.shape)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return ReshapeFn.backward(self.fcache, dy)
