"""module base module"""


from __future__ import annotations
from typing import Callable
from abc import ABC, abstractmethod

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, NumpyArray


__all__ = ["Module"]


class ModuleCompilationError(Exception):
    """Error with the compiling of the module."""


class Module(ABC):
    """Module base class."""

    def __init__(self) -> None:
        self.y: Tensor = tu.empty()
        self.parameters: list[Tensor] = []
        self.backward: Callable[[NumpyArray], NumpyArray | None] = lambda x: None
        self.training: bool = False

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + " (Params: "
            + str(sum([p.data.size for p in self.parameters]))
            + ")"
        )

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

    def reset_grads(self):
        """Resets parameter grads to improve memory usage."""
        for p in self.parameters:
            p.reset_grads()
        self.y.reset_grads()

    def set_y(self, y: Tensor) -> None:
        self.y.data = y.data

    def set_y_grad(self, y_grad: NumpyArray) -> None:
        self.y.grad = y_grad
