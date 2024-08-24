"""Neural network reshaping modules."""

from typing import Optional

from ...tensor_ops.reshaping import moveaxis, reshape
from ...tensors import ShapeLike, Tensor
from .module import Module

__all__ = ["Reshape", "Flatten", "Moveaxis"]


class Reshape(Module):
    """Reshapes a tensor to fit a given shape.

    Parameters
    ----------
    output_shape : _ShapeLike
        The output's target shape not including the batch dimension.
    label: str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(self, output_shape: ShapeLike, label: Optional[str] = None) -> None:
        super().__init__(label)
        self.output_shape = output_shape

    def forward(self, x: Tensor) -> Tensor:
        y = reshape(x, (x.shape[0],) + self.output_shape)

        if self._is_training:
            self._backward = lambda dy: reshape(dy, x.shape)

        return y


class Flatten(Module):
    """Flatten layer used to flatten tensors not including the batch dimension.

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def forward(self, x: Tensor) -> Tensor:
        y = reshape(x, (x.shape[0], -1))

        if self._is_training:
            self._backward = lambda dy: reshape(dy, x.shape)

        return y


class Moveaxis(Module):
    """Reshapes a tensor to fit a given shape.

    Parameters
    ----------
    from_axis : int
        Original position of the axis to move.
    to_axis : int
        Destination position.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(
        self, from_axis: int, to_axis: int, label: Optional[str] = None
    ) -> None:
        super().__init__(label)
        self.from_axis = from_axis
        self.to_axis = to_axis

    def forward(self, x: Tensor) -> Tensor:
        y = moveaxis(x, self.from_axis, self.to_axis)

        if self._is_training:
            self._backward = lambda dy: moveaxis(dy, self.from_axis, self.to_axis)

        return y
