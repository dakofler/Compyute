"""module base module"""


from __future__ import annotations
from typing import Callable
from abc import ABC, abstractmethod

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, NumpyArray, ShapeError


__all__ = ["Module"]


class Module(ABC):
    """Module base class."""

    def __init__(self) -> None:
        """Module base class."""
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
        """Performs a forward pass through the module.

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
        """Saves the module output to y tensor.

        Parameters
        ----------
        y : Tensor
            Module output tensor.
        """
        self.y.data = y.data

    def set_y_grad(self, y_grad: NumpyArray) -> None:
        """Saves the module output gradients to y tensor.

        Parameters
        ----------
        y_grad : NumpyArray
            Module output tensor gradients.
        """
        if y_grad.shape != self.y.shape:
            raise ShapeError(
                f"Grad shape {y_grad.shape} does not match y shape {self.y.shape}"
            )
        self.y.grad = y_grad
