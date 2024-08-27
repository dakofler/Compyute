"""Neural network reshaping modules."""

from typing import Optional

from ...tensors import ShapeLike, Tensor
from ..functional.reshapes import Fflatten, FReshape
from .module import Module

__all__ = ["Flatten", "Reshape"]


class Flatten(Module):
    """Flattes tensors not including the batch dimension.

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def forward(self, x: Tensor) -> Tensor:
        return Fflatten.forward(self._fcache, x)

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        return Fflatten.backward(self._fcache, dy)


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

    def forward(self, x: Tensor) -> Tensor:
        return FReshape.forward(self._fcache, x, self.shape)

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        return FReshape.backward(self._fcache, dy)
