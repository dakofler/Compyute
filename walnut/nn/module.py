"""module base module"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from abc import ABC, abstractmethod

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, NumpyArray, ShapeLike


__all__ = ["Module"]


class ModuleCompilationError(Exception):
    """Error with the compiling of the module."""


@dataclass(repr=False, init=False)
class Module(ABC):
    """Module base class."""

    def __init__(self, input_shape: ShapeLike | None = None) -> None:
        self.input_shape = input_shape
        self.x: Tensor = tu.empty()
        self.y: Tensor = tu.empty()
        self.parameters: list[Tensor] = []
        self.backward: Callable[[NumpyArray], NumpyArray] | None = None
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
    def __call__(self, x: Tensor) -> Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        ----------
        Tensor
            Computed Output.
        """

    def get_parameter_count(self) -> int:
        """Returns the total number of trainable parameters of the module."""
        return 0

    def set_x(self, x: Tensor) -> None:
        self.x.data = x.data

    def set_x_grad(self, x_grad: NumpyArray) -> None:
        self.x.grad = x_grad

    def set_y(self, y: Tensor) -> None:
        self.y.data = y.data

    def set_y_grad(self, y_grad: NumpyArray) -> None:
        self.y.grad = y_grad
