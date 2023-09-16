"""Tensor reshaping layers module"""

from __future__ import annotations

from walnut.tensor import Tensor, ArrayLike, ShapeLike
from walnut.nn.module import Module


__all__ = ["Reshape", "Flatten", "Moveaxis"]


class Reshape(Module):
    """Flatten layer used to reshape tensors to any shape."""

    def __init__(self, output_shape: ShapeLike) -> None:
        """Reshapes a tensor to fit a given shape.

        Parameters
        ----------
        output_shape : ShapeLike
            The output's target shape..
        """
        super().__init__()
        self.output_shape = output_shape

    def __repr__(self) -> str:
        name = self.__class__.__name__
        output_shape = self.output_shape
        return f"{name}({output_shape=})"

    def __call__(self, x: Tensor) -> Tensor:
        y = x.reshape(self.output_shape)

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                return dy.reshape(x.shape)

            self.backward = backward

        self.set_y(y)
        return y


class Flatten(Module):
    """Flatten layer used to reshape tensors to shape (b, -1)."""

    def __call__(self, x: Tensor) -> Tensor:
        y = x.reshape((x.shape[0], -1))

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                return dy.reshape(x.shape)

            self.backward = backward

        self.set_y(y)
        return y


class Moveaxis(Module):
    """Moveaxis layer used to swap tensor dimensions."""

    def __init__(self, from_axis: int, to_axis: int) -> None:
        """Reshapes a tensor to fit a given shape.

        Parameters
        ----------
        from_axis : int
            Original positions of the axes to move. These must be unique.
        to_axis : int
            Destination positions for each of the original axes. These must also be unique.
        """
        super().__init__()
        self.from_axis = from_axis
        self.to_axis = to_axis

    def __repr__(self) -> str:
        name = self.__class__.__name__
        from_axis = self.from_axis
        to_axis = self.to_axis
        return f"{name}({from_axis=}, {to_axis=})"

    def __call__(self, x: Tensor) -> Tensor:
        y = x.moveaxis(self.from_axis, self.to_axis)

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)

                return (
                    Tensor(dy, device=self.device)
                    .moveaxis(self.to_axis, self.from_axis)
                    .data
                )

            self.backward = backward

        self.set_y(y)
        return y
