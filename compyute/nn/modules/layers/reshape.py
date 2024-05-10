"""Tensor reshaping layers module"""

from typing import Optional
from ..module import Module
from ....basetensor import Tensor
from ....types import ShapeLike


__all__ = ["Reshape", "Flatten", "Moveaxis"]


class Reshape(Module):
    """Flatten layer used to reshape tensors to any shape."""

    def __init__(self, output_shape: ShapeLike, label: Optional[str] = None) -> None:
        """Reshapes a tensor to fit a given shape.

        Parameters
        ----------
        output_shape : ShapeLike
            The output's target shape not including the batch dimension.
        label: str, optional
            Module label.
        """
        super().__init__(label)
        self.output_shape = output_shape

    def forward(self, x: Tensor) -> Tensor:
        y = x.reshape((x.shape[0],) + self.output_shape)

        if self.training:
            self.backward_fn = lambda dy: dy.reshape(x.shape)

        return y


class Flatten(Module):
    """Flatten layer used to flatten tensors not including the batch dimension."""

    def forward(self, x: Tensor) -> Tensor:
        y = x.reshape((x.shape[0], -1))

        if self.training:
            self.backward_fn = lambda dy: dy.reshape(x.shape)

        return y


class Moveaxis(Module):
    """Moveaxis layer used to swap tensor dimensions."""

    def __init__(self, from_axis: int, to_axis: int, label: Optional[str] = None) -> None:
        """Reshapes a tensor to fit a given shape.

        Parameters
        ----------
        from_axis : int
            Original positions of the axes to move. These must be unique.
        to_axis : int
            Destination positions for each of the original axes. These must also be unique.
        label: str, optional
            Module label.
        """
        super().__init__(label)
        self.from_axis = from_axis
        self.to_axis = to_axis

    def forward(self, x: Tensor) -> Tensor:
        y = x.moveaxis(self.from_axis, self.to_axis)

        if self.training:
            self.backward_fn = lambda dy: dy.moveaxis(self.to_axis, self.from_axis)

        return y
