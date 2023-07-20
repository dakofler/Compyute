"""utility modules module"""


from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, NumpyArray, ShapeLike, AxisLike
from walnut.nn.modules.module import Module


__all__ = ["MaxPooling", "Reshape", "Moveaxis", "Dropout"]


@dataclass(init=False, repr=False)
class MaxPooling(Module):
    """MaxPoling module used to reduce information to avoid overfitting."""

    def __init__(
        self,
        p_window: tuple[int, int] = (2, 2),
        input_shape: ShapeLike | None = None,
    ) -> None:
        """MaxPoling module used to reduce information to avoid overfitting.

        Parameters
        ----------
        p_window : tuple[int, int], optional
             Shape of the pooling window used for the pooling operation, by default (2, 2)
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the module is used as input, by default None
        """
        super().__init__(input_shape=input_shape)
        self.p_window = p_window
        self.p_map: NumpyArray = np.empty(0, dtype="float32")

    def __call__(self, x: Tensor) -> Tensor:
        super().__call__(x)
        # cut off values to fit the pooling window
        y_fit = self.x.shape[2] // self.p_window[0] * self.p_window[0]
        x_fit = self.x.shape[3] // self.p_window[1] * self.p_window[1]
        x_crop = self.x[:, :, :y_fit, :x_fit]

        # init output as zeros (b, c, y, k)
        p_y, p_x = self.p_window
        x_b, x_c, _, _ = self.x.shape
        self.y.data = tu.zeros(
            (x_b, x_c, x_crop.shape[2] // p_y, x_crop.shape[3] // p_x)
        ).data
        self.p_map = tu.zeros_like(x_crop).data
        for y in range(self.y.shape[2]):
            for x in range(self.y.shape[3]):
                chunk = self.x.data[
                    :, :, y * p_y : (y + 1) * p_y, x * p_x : (x + 1) * p_x
                ]
                self.y.data[:, :, y, x] = np.max(chunk, axis=(2, 3))
        y_s = self.__stretch(self.y.data, self.p_window, (2, 3), x_crop.shape)
        self.p_map = (x_crop.data == y_s) * 1.0
        return self.y

    def backward(self, y_grad: NumpyArray) -> NumpyArray:
        super().backward(y_grad)
        dy_s = self.__stretch(self.y.grad, self.p_window, (2, 3), self.p_map.shape)
        # use p_map as mask for grads
        self.x.grad = np.resize((dy_s * self.p_map), self.x.shape)
        return self.x.grad

    def __stretch(
        self,
        x: NumpyArray,
        streching: tuple[int, int],
        axis: tuple[int, int],
        target_shape: ShapeLike,
    ) -> NumpyArray:
        fa1, fa2 = streching
        ax1, ax2 = axis
        x_stretched = np.repeat(x, fa1, axis=ax1)
        x_stretched = np.repeat(x_stretched, fa2, axis=ax2)
        # resize to fit target shape by filling with zeros
        return np.resize(x_stretched, target_shape)


@dataclass(init=False, repr=False)
class Reshape(Module):
    """Flatten module used to reshape tensors to shape (b, c_out)."""

    def __init__(
        self, output_shape: ShapeLike = (-1,), input_shape: ShapeLike | None = None
    ) -> None:
        """Reshapes a tensor to fit a given shape.

        Parameters
        ----------
        output_shape : ShapeLike, optional
            The output's target shape, by default (-1,).
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the module is used as input, by default None.
        """
        super().__init__(input_shape=input_shape)
        self.output_shape = output_shape

    def __call__(self, x: Tensor) -> Tensor:
        super().__call__(x)
        self.y.data = self.x.data.reshape((self.x.shape[0], *self.output_shape))
        return self.y

    def backward(self, y_grad: NumpyArray) -> NumpyArray:
        super().backward(y_grad)
        self.x.grad = self.y.grad.reshape(self.x.shape)
        return self.x.grad


@dataclass(init=False, repr=False)
class Moveaxis(Module):
    """Moveaxis module used to swap tensor dimensions."""

    def __init__(
        self,
        axis_from: AxisLike,
        axis_to: AxisLike,
        input_shape: ShapeLike | None = None,
    ) -> None:
        """Reshapes a tensor to fit a given shape.

        Parameters
        ----------
        axis_from : AxisLike
            What axis to move.
        axis_to : AxisLike
            Where to move the axis.
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the module is used as input, by default None.
        """
        super().__init__(input_shape=input_shape)
        self.axis_from = axis_from
        self.axis_to = axis_to

    def __call__(self, x: Tensor) -> Tensor:
        super().__call__(x)
        self.y.data = np.moveaxis(self.x.data, self.axis_from, self.axis_to)
        return self.y

    def backward(self, y_grad: NumpyArray) -> NumpyArray:
        super().backward(y_grad)
        self.x.grad = np.moveaxis(self.y.grad, self.axis_to, self.axis_from)
        return self.x.grad


@dataclass(init=False, repr=False)
class Dropout(Module):
    """Dropout module used to randomly reduce information and avoid overfitting."""

    def __init__(self, d_rate: float, input_shape: ShapeLike | None = None) -> None:
        """Dropout module used to randomly reduce information and avoid overfitting.

        Parameters
        ----------
        d_rate : float
            Probability of values being set to 0.
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the module is used as input, by default None.
        """
        super().__init__(input_shape=input_shape)
        self.d_rate = d_rate
        self.d_map: NumpyArray = np.empty(0, dtype="float32")

    def __call__(self, x: Tensor) -> Tensor:
        super().__call__(x)
        if not self.training:
            self.y.data = self.x.data
        else:
            drop_rate = self.d_rate
            d_map = np.random.choice([0, 1], self.x.shape, p=[drop_rate, 1 - drop_rate])
            self.d_map = d_map.astype("float32")
            self.y.data = self.x.data * self.d_map / (1.0 - drop_rate)
        return self.y

    def backward(self, y_grad: NumpyArray) -> NumpyArray:
        super().backward(y_grad)
        # use d_map as mask for grads
        self.x.grad = self.y.grad * self.d_map / (1.0 - self.d_rate)
        return self.x.grad
