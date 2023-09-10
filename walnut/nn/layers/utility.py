"""utility layers layer"""

from __future__ import annotations

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ArrayLike, ShapeLike
from walnut.nn.module import Module
from walnut.nn.funcional import stretch2d


__all__ = ["MaxPooling2d", "Reshape", "Flatten", "Moveaxis", "Dropout"]


class MaxPooling2d(Module):
    """MaxPoling layer used to reduce information to avoid overfitting."""

    def __init__(self, kernel_size: tuple[int, int] = (2, 2)) -> None:
        """MaxPoling layer used to reduce information to avoid overfitting.

        Parameters
        ----------
        kernel_size : tuple[int, int], optional
             Shape of the pooling window used for the pooling operation, by default (2, 2).
        """
        super().__init__()
        self.kernel_size = kernel_size

    def __repr__(self) -> str:
        name = self.__class__.__name__
        kernel_size = self.kernel_size
        return f"{name}({kernel_size=})"

    def __call__(self, x: Tensor) -> Tensor:
        # cut off values to fit the pooling window
        y_fit = x.shape[-2] // self.kernel_size[0] * self.kernel_size[0]
        x_fit = x.shape[-1] // self.kernel_size[1] * self.kernel_size[1]
        x_crop = x[:, :, :y_fit, :x_fit]

        p_y, p_x = self.kernel_size
        x_b, x_c, _, _ = x.shape
        _, _, xc_y, xc_x = x_crop.shape
        y = tu.zeros(
            (x_b, x_c, xc_y // p_y, xc_x // p_x),
            device=self.device,
        )
        for yi in range(y.shape[-2]):
            for xi in range(y.shape[-1]):
                cnk = x.data[:, :, yi * p_y : (yi + 1) * p_y, xi * p_x : (xi + 1) * p_x]
                y[:, :, yi, xi] = cnk.max(axis=(-2, -1))

        y_s = stretch2d(y, self.kernel_size, x_crop.shape)
        p_map = (x_crop == y_s) * 1.0

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)

                dy_s = stretch2d(
                    Tensor(dy, device=self.device), self.kernel_size, p_map.shape
                )
                # use p_map as mask for grads
                return (dy_s * p_map).resize(x.shape).data

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


class Dropout(Module):
    """Dropout layer used to randomly reduce information and avoid overfitting."""

    def __init__(self, p: float = 0.5) -> None:
        """Dropout layer used to randomly reduce information and avoid overfitting.

        Parameters
        ----------
        p : float, optional
            Probability of values being set to zero, by default 0.5.
        """
        super().__init__()
        self.p = p

    def __repr__(self) -> str:
        name = self.__class__.__name__
        p = self.p
        return f"{name}({p=})"

    def __call__(self, x: Tensor) -> Tensor:
        if self.training:
            choices = Tensor([0, 1])
            probs = Tensor([self.p, 1.0 - self.p])
            d_map = tu.random_choice(choices, probs, x.shape, self.device)
            y = x * d_map / (1.0 - self.p)

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                # use d_map as mask for grads
                return dy * d_map.data / (1.0 - self.p)

            self.backward = backward

        else:
            y = x

        self.set_y(y)
        return y
