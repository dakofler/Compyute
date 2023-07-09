"""utility modules module"""


from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, NumpyArray, ShapeLike


class ModuleCompilationError(Exception):
    """Error with the compiling of the module."""


@dataclass(repr=False, init=False)
class Module(ABC):
    """Module base class."""

    def __init__(self, input_shape: ShapeLike | None = None):
        self.input_shape = input_shape
        self.x: Tensor = Tensor()
        self.y: Tensor = Tensor()
        self.parameters: list[Tensor] | None = None
        self.compiled: bool = False
        self.training: bool = False

    def __repr__(self) -> str:
        name = self.__class__.__name__
        if not self.compiled:
            return name
        x_shape = str(self.x.shape[1:])
        w_shape = b_shape = "(,)"
        y_shape = str(self.y.shape[1:])
        return (
            f"{name:15s} | {x_shape:15s} | {w_shape:15s} | "
            + f"{b_shape:15s} | {y_shape:15s} | 0"
        )

    def compile(self) -> None:
        """Connects modules within a model."""
        if self.input_shape is not None:
            self.x = tu.ones((1, *self.input_shape))
        self.compiled = True

    @abstractmethod
    def forward(self) -> None:
        """Performs a forward pass ."""

    @abstractmethod
    def backward(self) -> None:
        """Performs a backward pass and computes gradients."""

    def get_parameter_count(self) -> int:
        """Returns the total number of trainable parameters of the module."""
        return 0


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

    def forward(self) -> None:
        # init output as zeros (b, c, y, k)
        x_crop = self.__crop()
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

    def backward(self) -> None:
        dy_s = self.__stretch(self.y.grad, self.p_window, (2, 3), self.p_map.shape)
        # use p_map as mask for grads
        self.x.grad = np.resize((dy_s * self.p_map), self.x.shape)

    def __crop(self) -> Tensor:
        w_y, w_x = self.p_window
        _, _, x_y, x_x = self.x.shape
        y_fit = x_y // w_y * w_y
        x_fit = x_x // w_x * w_x
        return self.x[:, :, :y_fit, :x_fit]

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

    def forward(self) -> None:
        self.y.data = self.x.data.reshape((self.x.shape[0], *self.output_shape))

    def backward(self) -> None:
        self.x.grad = self.y.grad.reshape(self.x.shape)


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

    def forward(self) -> None:
        if not self.training:
            self.y.data = self.x.data
        else:
            drop_rate = self.d_rate
            d_map = np.random.choice([0, 1], self.x.shape, p=[drop_rate, 1 - drop_rate])
            self.d_map = d_map.astype("float32")
            self.y.data = self.x.data * self.d_map / (1.0 - drop_rate)

    def backward(self) -> None:
        # use d_map as mask for grads
        self.x.grad = self.y.grad * self.d_map / (1.0 - self.d_rate)
