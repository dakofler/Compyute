"""utility layers layer"""


from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, NumpyArray, ShapeLike
from walnut.nn.module import Module


__all__ = ["MaxPooling2d", "Reshape", "Moveaxis", "Dropout"]


@dataclass(init=False, repr=False)
class MaxPooling2d(Module):
    """MaxPoling layer used to reduce information to avoid overfitting."""

    def __init__(
        self,
        kernel_size: tuple[int, int] = (2, 2),
        input_shape: ShapeLike | None = None,
    ) -> None:
        """MaxPoling layer used to reduce information to avoid overfitting.

        Parameters
        ----------
        kernel_size : tuple[int, int], optional
             Shape of the pooling window used for the pooling operation, by default (2, 2)
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the layer is used as input, by default None
        """
        super().__init__(input_shape=input_shape)
        self.kernel_size = kernel_size

    def __call__(self, x: Tensor) -> Tensor:
        # cut off values to fit the pooling window
        y_fit = x.shape[-2] // self.kernel_size[0] * self.kernel_size[0]
        x_fit = x.shape[-1] // self.kernel_size[1] * self.kernel_size[1]
        x_crop = x[:, :, :y_fit, :x_fit]

        p_y, p_x = self.kernel_size
        x_b, x_c, _, _ = x.shape
        y = tu.zeros((x_b, x_c, x_crop.shape[-2] // p_y, x_crop.shape[-1] // p_x))
        for yi in range(y.shape[-2]):
            for xi in range(y.shape[-1]):
                cnk = x.data[:, :, yi * p_y : (yi + 1) * p_y, xi * p_x : (xi + 1) * p_x]
                y[:, :, yi, xi] = cnk.max(axis=(-2, -1))

        y_s = tu.stretch(y, self.kernel_size, x_crop.shape)
        p_map = (x_crop == y_s) * 1.0

        if self.training:

            def backward(y_grad: NumpyArray) -> NumpyArray:
                dy_s = tu.stretch(Tensor(y_grad), self.kernel_size, p_map.shape)
                # use p_map as mask for grads
                x_grad = np.resize((dy_s * p_map).data, x.shape)

                self.set_y_grad(y_grad)
                self.set_x_grad(x_grad)
                return x_grad

            self.backward = backward

        self.set_x(x)
        self.set_y(y)
        return y


@dataclass(init=False, repr=False)
class Reshape(Module):
    """Flatten layer used to reshape tensors to shape (b, c_out)."""

    def __init__(
        self, output_shape: ShapeLike = (-1,), input_shape: ShapeLike | None = None
    ) -> None:
        """Reshapes a tensor to fit a given shape.

        Parameters
        ----------
        output_shape : ShapeLike, optional
            The output's target shape, by default (-1,).
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the layer is used as input, by default None.
        """
        super().__init__(input_shape=input_shape)
        self.output_shape = output_shape

    def __call__(self, x: Tensor) -> Tensor:
        y = x.reshape((x.shape[0], *self.output_shape))

        if self.training:

            def backward(y_grad: NumpyArray) -> NumpyArray:
                x_grad = y_grad.reshape(x.shape)

                self.set_y_grad(y_grad)
                self.set_x_grad(x_grad)
                return x_grad

            self.backward = backward

        self.set_x(x)
        self.set_y(y)
        return y


@dataclass(init=False, repr=False)
class Moveaxis(Module):
    """Moveaxis layer used to swap tensor dimensions."""

    def __init__(
        self,
        from_axis: int,
        to_axis: int,
        input_shape: ShapeLike | None = None,
    ) -> None:
        """Reshapes a tensor to fit a given shape.

        Parameters
        ----------
        from_axis : int
            Original positions of the axes to move. These must be unique.
        to_axis : int
            Destination positions for each of the original axes. These must also be unique.
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the layer is used as input, by default None.
        """
        super().__init__(input_shape=input_shape)
        self.from_axis = from_axis
        self.to_axis = to_axis

    def __call__(self, x: Tensor) -> Tensor:
        y = x.moveaxis(self.from_axis, self.to_axis)

        if self.training:

            def backward(y_grad: NumpyArray) -> NumpyArray:
                x_grad = np.moveaxis(y_grad, self.to_axis, self.from_axis)

                self.set_y_grad(y_grad)
                self.set_x_grad(x_grad)
                return x_grad

            self.backward = backward

        self.set_x(x)
        self.set_y(y)
        return y


@dataclass(init=False, repr=False)
class Dropout(Module):
    """Dropout layer used to randomly reduce information and avoid overfitting."""

    def __init__(self, p: float = 0.5, input_shape: ShapeLike | None = None) -> None:
        """Dropout layer used to randomly reduce information and avoid overfitting.

        Parameters
        ----------
        p : float, optional
            Probability of values being set to zero, by default 0.5.
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the layer is used as input, by default None.
        """
        super().__init__(input_shape=input_shape)
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if not self.training:
            y = x
        else:
            d_map = np.random.choice([0.0, 1.0], x.shape, p=[self.p, 1.0 - self.p])
            y = x * d_map / (1.0 - self.p)

            def backward(y_grad: NumpyArray) -> NumpyArray:
                # use d_map as mask for grads
                x_grad = y_grad * d_map.data / (1.0 - self.p)

                self.set_y_grad(y_grad)
                self.set_x_grad(x_grad)
                return x_grad

            self.backward = backward

        self.set_x(x)
        self.set_y(y)
        return y
