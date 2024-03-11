"""Tensor reshaping layers module"""

from compyute.functional import zeros_like
from compyute.nn.module import Module
from compyute.tensor import Tensor, ShapeLike, ArrayLike


__all__ = ["Slice", "Reshape", "Flatten", "Moveaxis"]


class Slice(Module):
    """Slices a tensor."""

    def __init__(self, s: list[None | int | slice]) -> None:
        """Slices a tensor.

        Parameters
        ----------
        s : list[None, int, slice]
            Slice applied to a tensor not including the batch dimension.
            e.g. [slice(None), 1] is equivalent to [:, 1]
        """
        super().__init__()
        self.s = s

    def __repr__(self) -> str:
        name = self.__class__.__name__
        s = self.s
        return f"{name}({s=})"

    def forward(self, x: Tensor) -> Tensor:
        s = [slice(None)] + self.s
        y = x[*s]

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                dx = zeros_like(x, device=self.device, dtype=dy.dtype).data
                dx[*s] = dy
                return dx

            self.backward = backward

        self.set_y(y)
        return y


class Reshape(Module):
    """Flatten layer used to reshape tensors to any shape."""

    def __init__(self, output_shape: ShapeLike) -> None:
        """Reshapes a tensor to fit a given shape.

        Parameters
        ----------
        output_shape : ShapeLike
            The output's target shape not including the batch dimension.
        """
        super().__init__()
        self.output_shape = output_shape

    def __repr__(self) -> str:
        name = self.__class__.__name__
        output_shape = self.output_shape
        return f"{name}({output_shape=})"

    def forward(self, x: Tensor) -> Tensor:
        y = x.reshape((x.shape[0],) + self.output_shape)

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                return dy.reshape(x.shape)

            self.backward = backward

        self.set_y(y)
        return y


class Flatten(Module):
    """Flatten layer used to flatten tensors not including the batch dimension."""

    def forward(self, x: Tensor) -> Tensor:
        y = x.reshape((x.shape[0], -1))

        if self.training:
            self.backward = lambda dy: dy.reshape(x.shape)

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

    def forward(self, x: Tensor) -> Tensor:
        y = x.moveaxis(self.from_axis, self.to_axis)

        if self.training:
            self.backward = (
                lambda dy: Tensor(dy, dtype=dy.dtype, device=self.device)
                .moveaxis(self.to_axis, self.from_axis)
                .data
            )

        return y
