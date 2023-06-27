"""utility layers module"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

from walnut import tensor
from walnut.tensor import Tensor


class LayerCompilationError(Exception):
    """Error with the compiling of the layer."""


@dataclass(repr=False, init=False)
class Layer(ABC):
    """Layer base class."""

    def __init__(self, input_shape: tuple[int, ...] | None = None):
        self.input_shape = input_shape
        self.x: Tensor = Tensor()
        self.y: Tensor = Tensor()
        self.compiled: bool = False

    def __repr__(self) -> str:
        name = self.__class__.__name__
        if not self.compiled:
            return name
        x_shape = str(self.x.shape[1:])  # x is never none here if layer is compiled
        w_shape = b_shape = "(,)"
        y_shape = str(self.y.shape[1:])  # y is never none here if layer is compiled
        return (
            f"{name:15s} | {x_shape:15s} | {w_shape:15s} | "
            + f"{b_shape:15s} | {y_shape:15s} | 0"
        )

    def compile(self) -> None:
        """Connects layers within a model."""
        if self.input_shape is not None:
            self.x = tensor.ones((1, *self.input_shape))
        self.compiled = True

    @abstractmethod
    def forward(self, mode: str = "eval") -> None:
        """Performs a forward pass ."""

    @abstractmethod
    def backward(self) -> None:
        """Performs a backward pass and computes gradients."""

    def get_parameter_count(self) -> int:
        """Returns the total number of trainable parameters of the layer."""
        return 0


@dataclass(init=False, repr=False)
class MaxPooling(Layer):
    """MaxPoling layer used to reduce information to avoid overfitting."""

    def __init__(
        self,
        p_window: tuple[int, int] = (2, 2),
        input_shape: tuple[int, ...] | None = None,
    ) -> None:
        """MaxPoling layer used to reduce information to avoid overfitting.

        Parameters
        ----------
        p_window : tuple[int, int], optional
             Shape of the pooling window used for the pooling operation, by default (2, 2)
        input_shape : tuple[int, ...] | None, optional
            Shape of a sample. Required if the layer is used as input, by default None
        """
        super().__init__(input_shape=input_shape)
        self.p_window = p_window
        self.p_map: npt.NDArray[Any] = np.empty(0, dtype="float32")

    def forward(self, mode: str = "eval") -> None:
        # init output as zeros (b, c, y, k)
        x_crop = self.__crop()
        p_y, p_x = self.p_window
        x_b, x_c, _, _ = self.x.shape
        self.y.data = tensor.zeros(
            (x_b, x_c, x_crop.shape[2] // p_y, x_crop.shape[3] // p_x)
        ).data
        self.p_map = tensor.zeros_like(x_crop).data
        for y in range(self.y.shape[2]):
            for x in range(self.y.shape[3]):
                chunk = self.x.data[
                    :, :, y * p_y : (y + 1) * p_y, x * p_x : (x + 1) * p_x
                ]
                self.y.data[:, :, y, x] = np.max(chunk, axis=(2, 3))
        y_s = self.__stretch(self.y.data, self.p_window, (2, 3), x_crop.shape)
        self.p_map = (x_crop.data == y_s) * 1.0

    def backward(self) -> None:
        dy_s = self.__stretch(self.y.grad, self.p_window, (2, 3), self.p_map.shape)
        _, _, x_y, x_x = self.x.shape
        self.x.grad = (dy_s * self.p_map)[
            :, :, :x_y, :x_x
        ]  # use p_map as mask for grads

    def __crop(self) -> Tensor:
        w_y, w_x = self.p_window
        _, _, x_y, x_x = self.x.shape
        y_fit = x_y // w_y * w_y
        x_fit = x_x // w_x * w_x
        return self.x[:, :, :y_fit, :x_fit]

    def __stretch(
        self,
        x: npt.NDArray[Any],
        streching: tuple[int, int],
        axis: tuple[int, int],
        target_shape: tuple[int, ...],
    ) -> npt.NDArray[Any]:
        fa1, fa2 = streching
        ax1, ax2 = axis
        x_stretched = np.repeat(x, fa1, axis=ax1)
        x_stretched = np.repeat(x_stretched, fa2, axis=ax2)
        return np.resize(x_stretched, target_shape)


class Flatten(Layer):
    """Flatten layer used to reshape tensors to shape (b, c_out)."""

    def __init__(self, input_shape: tuple[int, ...] | None = None) -> None:
        """Flatten layer used to reshape tensors to shape (b, c_out).

        Parameters
        ----------
        input_shape : tuple[int, ...] | None, optional
            Shape of a sample. Required if the layer is used as input, by default None.
        """
        super().__init__(input_shape=input_shape)

    def forward(self, mode: str = "eval") -> None:
        self.y.data = self.x.data.reshape(self.x.shape[0], -1)

    def backward(self) -> None:
        self.x.grad = np.resize(self.y.grad, self.x.shape)


@dataclass(init=False, repr=False)
class Dropout(Layer):
    """Dropout layer used to randomly reduce information and avoid overfitting."""

    def __init__(
        self, d_rate: float, input_shape: tuple[int, ...] | None = None
    ) -> None:
        """Dropout layer used to randomly reduce information and avoid overfitting.

        Parameters
        ----------
        d_rate : float
            Probability of values being set to 0.
        input_shape : tuple[int, ...] | None, optional
            Shape of a sample. Required if the layer is used as input, by default None.
        """
        super().__init__(input_shape=input_shape)
        self.d_rate = d_rate
        self.d_map: npt.NDArray[Any] = np.empty(0, dtype="float32")

    def forward(self, mode: str = "eval") -> None:
        if mode == "eval":
            self.y.data = self.x.data
        else:
            drop_rate = self.d_rate
            d_map = np.random.choice([0, 1], self.x.shape, p=[drop_rate, 1 - drop_rate])
            self.d_map = d_map.astype("float32")
            self.y.data = self.x.data * self.d_map / (1.0 - drop_rate)

    def backward(self) -> None:
        # use d_map as mask for grads
        self.x.grad = self.y.grad * self.d_map / (1.0 - self.d_rate)
