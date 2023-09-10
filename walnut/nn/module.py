"""module base module"""

from __future__ import annotations
from typing import Callable
from abc import ABC, abstractmethod

from walnut.tensor import Tensor, ArrayLike
import walnut.tensor_utils as tu


__all__ = ["Module"]


class Module(ABC):
    """Module base class."""

    def __init__(self) -> None:
        """Module base class."""
        self.y: Tensor = tu.empty()
        self.backward: Callable[[ArrayLike], ArrayLike] | None = None
        self._sub_modules: list[Module] = []
        self._parameters: list[Tensor] = []
        self._training: bool = False
        self._keep_output: bool = False
        self._device: str = "cpu"

    @property
    def sub_modules(self) -> list[Module]:
        """Trainable module parameters."""
        return self._sub_modules

    @sub_modules.setter
    def sub_modules(self, value: list[Module]) -> None:
        self._sub_modules = value

    @property
    def parameters(self) -> list[Tensor]:
        """Trainable module parameters."""
        p = self._parameters.copy()
        for module in self.sub_modules:
            p += module.parameters
        return p

    @parameters.setter
    def parameters(self, value: list[Tensor]) -> None:
        self._parameters = value

    @property
    def device(self) -> str:
        """Storage device."""
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        if value not in ("cpu", "cuda"):
            raise ValueError("Unknown device.")
        self._device = value

    def to_device(self, device: str) -> None:
        """Moves the tensor to a specified device.

        Parameters
        ----------
        device : str
            Device to move the tensor to. Valid options are "cpu" and "cuda".
        """
        self._device = device
        for p in self.parameters:
            p.to_device(device)
        for module in self.sub_modules:
            module.to_device(device)

    @property
    def training(self) -> bool:
        """Puts the module in training mode.
        The forward behaviour might differ when in training mode."""
        return self._training

    @training.setter
    def training(self, value: bool) -> None:
        self._training = value
        for module in self.sub_modules:
            module.training = value

    @property
    def keep_output(self) -> bool:
        """Whether to keep output valuesduring forward and gradients during backward passes."""
        return self._keep_output

    @keep_output.setter
    def keep_output(self, value: bool) -> None:
        self._keep_output = value
        for module in self.sub_modules:
            module.keep_output = value

    def __repr__(self) -> str:
        string = f"{self.__class__.__name__}()"
        for module in self.sub_modules:
            string += "\n" + module.__repr__()
        return string

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
            Computed module output.
        """

    def set_y(self, y: Tensor) -> None:
        """Saves the module output to y tensor.

        Parameters
        ----------
        y : Tensor
            Module output tensor.
        """
        if self.keep_output:
            self.y.data = y.data.copy()

    def set_dy(self, dy: ArrayLike) -> None:
        """Saves the module output gradients to y tensor.

        Parameters
        ----------
        dy : NpArrayLike
            Module output tensor gradients.
        """
        if self.keep_output:
            self.y.grad = dy.copy()

    def clean(self) -> None:
        """Cleanes up the module by resetting temporary values."""
        self.y = tu.empty()
        self.y.grad = None
        self.backward = None

        for module in self.sub_modules:
            module.clean()
