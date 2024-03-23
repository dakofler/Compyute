"""Neural network module base module"""

from __future__ import annotations
from abc import ABC
from typing import Callable
from ..parameter import Parameter
from ...tensor import Tensor, ShapeError
from ...types import DeviceLike


__all__ = ["Module"]


class Module(ABC):
    """Module base class."""

    def __init__(self) -> None:
        """Module base class."""
        self.backward: Callable[[Tensor], Tensor] | None = None
        self.y: Tensor | None = None
        self.__child_modules: list[Module] = []
        self.__device: DeviceLike = "cpu"
        self.__retain_values: bool = False
        self.__training: bool = False

    # ----------------------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------------------

    @property
    def child_modules(self) -> list[Module]:
        """Model child modules."""
        return self.__child_modules

    @child_modules.setter
    def child_modules(self, value: list[Module]) -> None:
        self.__child_modules = value

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

        for module in self.child_modules:
            module.to_device(device)

    @property
    def parameters(self) -> list[Parameter]:
        """Returns the list of module parameters."""
        p = [i[1] for i in self.__dict__.items() if isinstance(i[1], Parameter)]

        for module in self.child_modules:
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

        for module in self.child_modules:
            module.retain_values = value

    @property
    def training(self) -> bool:
        """Puts the module in training mode.
        The forward behaviour might differ when in training mode."""
        return self.__training

    @training.setter
    def training(self, value: bool) -> None:
        if self.__training == value:
            return
        self.__training = value

        for module in self.child_modules:
            module.training = value

    # ----------------------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        string = f"{self.__class__.__name__}()"
        for module in self.child_modules:
            string += "\n" + module.__repr__()
        return string

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

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
        if self.training:

            def backward(dy: Tensor) -> Tensor:
                self.set_dy(dy)
                return dy

            self.backward = backward

        self.set_y(x)
        return x

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
        self.backward = None

        for p in self.parameters:
            p.grad = None

        for module in self.child_modules:
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
