"""module base module"""

from __future__ import annotations
from typing import Callable
from abc import ABC, abstractmethod

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, NpArrayLike


__all__ = ["Module"]


class Module(ABC):
    """Module base class."""

    def __init__(self) -> None:
        """Module base class."""
        self.y: Tensor = tu.empty()
        self.parameters: list[Tensor] = []
        self.backward: Callable[[NpArrayLike], NpArrayLike] = lambda y_grad: y_grad
        self.training: bool = False

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}()"

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

    def get_parameters(self):
        """Returns parameters of the module and it's layers."""
        return self.parameters.copy()

    def reset_grads(self):
        """Resets parameter grads to improve memory usage."""
        for parameter in self.parameters:
            parameter.grad = None
        self.y.grad = None

    def training_mode(self):
        """Puts the module into training mode. Some modules may have different forward
        behaviour if in training mode. Backward behaviour is only defined in training mode.
        """
        self.training = True

    def eval_mode(self):
        """Puts the module into evaluation mode. Some modules may have different forward
        behaviour if in training mode. Backward behaviour is only defined in training mode.
        """
        self.training = False

    def set_y(self, y: Tensor) -> None:
        """Saves the module output to y tensor.

        Parameters
        ----------
        y : Tensor
            Module output tensor.
        """
        self.y.data = y.data.copy()

    def set_y_grad(self, y_grad: NpArrayLike) -> None:
        """Saves the module output gradients to y tensor.

        Parameters
        ----------
        y_grad : NumpyArray
            Module output tensor gradients.
        """
        self.y.grad = y_grad.copy()
