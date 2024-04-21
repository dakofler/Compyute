"""Neural network module base module"""

from __future__ import annotations
from abc import ABC
from typing import Callable, Optional
from ..parameter import Parameter
from ...tensor import Tensor, ShapeError
from ...types import DeviceLike


__all__ = ["Module", "Passthrough"]


class Module(ABC):
    """Module base class."""

    def __init__(self) -> None:
        """Module base class."""
        self.y: Optional[Tensor] = None
        self.backward_fn: Optional[Callable[[Tensor], Optional[Tensor]]] = None
        self.__modules: Optional[list[Module]] = None
        self.__device: DeviceLike = "cpu"
        self.__retain_values: bool = False
        self.__training: bool = False
        self.__trainable: bool = True

    # ----------------------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------------------

    @property
    def modules(self) -> Optional[list[Module]]:
        """List of child modules."""
        return self.__modules

    @modules.setter
    def modules(self, value: Optional[list[Module]]) -> None:
        self.__modules = value

    @property
    def device(self) -> str:
        """Device the module tensors are stored on."""
        return self.__device

    def to_device(self, device: DeviceLike) -> None:
        """Moves the tensor to a specified device.

        Parameters
        ----------
        device : DeviceLike
            Device to move the tensor to ("cuda" or "cpu").
        """
        if device == self.device:
            return
        self.__device = device

        if self.y is not None:
            self.y.to_device(device)

        for p in self.parameters:
            p.to_device(device)

        if self.modules is not None:
            for module in self.modules:
                module.to_device(device)

    @property
    def parameters(self) -> list[Parameter]:
        """Returns the list of module parameters."""
        p = [i[1] for i in self.__dict__.items() if isinstance(i[1], Parameter)]

        if self.modules is not None:
            for module in self.modules:
                p += module.parameters

        return p

    @property
    def retain_values(self) -> bool:
        """Sets the module to keep its outputs and gradients."""
        return self.__retain_values

    @retain_values.setter
    def retain_values(self, value: bool) -> None:
        if self.__retain_values == value:
            return

        self.__retain_values = value

        if self.modules is not None:
            for module in self.modules:
                module.retain_values = value

    @property
    def training(self) -> bool:
        """Puts the module in training mode.
        The forward behaviour might differ for some modules when in training mode."""
        return self.__training

    @training.setter
    def training(self, value: bool) -> None:
        if self.__training == value:
            return
        self.__training = value

        if self.modules is not None:
            for module in self.modules:
                module.training = value

    @property
    def trainable(self) -> bool:
        """Sets the module to not be trainable."""
        return self.__trainable

    @trainable.setter
    def trainable(self, value: bool) -> None:
        if self.__trainable == value:
            return
        self.__trainable = value

        if self.modules is not None:
            for module in self.modules:
                module.trainable = value

    # ----------------------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        string = f"{self.__class__.__name__}()"

        if self.modules is not None:
            for module in self.modules:
                string += "\n" + module.__repr__()

        return string

    def __call__(self, x: Tensor) -> Tensor:
        y = self.forward(x)
        self.set_y(y)
        return y

    # ----------------------------------------------------------------------------------------------
    # OTHER OPERATIONS
    # ----------------------------------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass through the module.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        ----------
        Tensor
            Computed module output.
        """
        return x

    def backward(self, dy: Tensor) -> Optional[Tensor]:
        """Performs a backward pass through the module.

        Parameters
        ----------
        dy : Tensor
            Output gradient tensor.

        Returns
        ----------
        Tensor, optional
            Input gradient tensor.
        """
        self.set_dy(dy)
        return dy if self.backward_fn is None else self.backward_fn(dy)

    def set_y(self, y: Tensor) -> None:
        """Saves the module output to y tensor.

        Parameters
        ----------
        y : Tensor
            Module output tensor.
        """
        if self.retain_values:
            self.y = y.copy()

    def set_dy(self, dy: Tensor) -> None:
        """Saves the module output gradients to y tensor.

        Parameters
        ----------
        dy : Tensor
            Module output tensor gradients.
        """
        if self.retain_values and self.y is not None:
            self.y.grad = dy.copy()

    def reset(self) -> None:
        """Resets temporary values like outputs and gradients."""
        self.y = None
        self.backward_fn = None

        for p in self.parameters:
            p.grad = None

        if self.modules is not None:
            for module in self.modules:
                module.reset()

    def check_dims(self, x: Tensor, valid_dims: list[int]) -> None:
        """Checks if a tensors dimensions match desired target dimensions.

        Parameters
        ----------
        x : Tensor
            Tensor whose dimensions are checked.
        valid_dims : int
            Valid numbers of dimension the tensor should have.

        Raises
        ------
        ShapeError
            If the tensor's dimensions do not match the target dimensions.
        """
        if x.ndim not in valid_dims:
            sender = self.__class__.__name__
            vdims = ", ".join([str(d) for d in valid_dims])
            raise ShapeError(
                f"{sender}: Number of input dimensions {
                    x.ndim} is not valid (valid: {vdims})"
            )

class Passthrough(Module):
    """Acts as a passthrough for data."""
