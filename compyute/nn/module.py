"""module base module"""

from __future__ import annotations
from typing import Callable
from abc import ABC

from compyute.nn.parameter import Parameter
from compyute.tensor import Tensor, ArrayLike
import compyute.tensor_functions as tf


__all__ = ["Module"]


class Module(ABC):
    """Module base class."""

    def __init__(self) -> None:
        """Module base class."""
        self.backward: Callable[[ArrayLike], ArrayLike] | None = None
        self._sub_modules: list[Module] = []
        self._remember: bool = False
        self.y: Tensor | None = None
        self._training: bool = False
        self._device: str = "cpu"

    @property
    def sub_modules(self) -> list[Module]:
        """List of sub-modules."""
        return self._sub_modules

    @sub_modules.setter
    def sub_modules(self, value: list[Module]) -> None:
        self._sub_modules = value

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
        if self.y is not None:
            self.y.to_device(device)

        for p in self.parameters():
            p.to_device(device)

        for module in self.sub_modules:
            module.to_device(device)

    @property
    def remember(self) -> bool:
        """Sets the module to keep its outputs and gradients."""
        return self._remember

    @remember.setter
    def remember(self, value: bool) -> None:
        self._remember = value
        self.y = tf.empty("float16", device=self.device) if value else None
        for module in self.sub_modules:
            module.remember = value

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

    def __repr__(self) -> str:
        string = f"{self.__class__.__name__}()"
        for module in self.sub_modules:
            string += "\n" + module.__repr__()
        return string

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

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

            def backward(dy: ArrayLike) -> ArrayLike:
                return dy

            self.backward = backward

        return x

    def set_y(self, y: Tensor) -> None:
        """Saves the module output to y tensor.

        Parameters
        ----------
        y : Tensor
            Module output tensor.
        """
        if self.remember and self.y is not None:
            self.y.data = y.data.copy()

    def set_dy(self, dy: ArrayLike) -> None:
        """Saves the module output gradients to y tensor.

        Parameters
        ----------
        dy : ArrayLike
            Module output tensor gradients.
        """
        if self.remember and self.y is not None:
            self.y.grad = dy.copy()

    def parameters(self) -> list[Parameter]:
        """Returns the list of module parameters."""
        parameters = []

        # get parameter objects
        for item in self.__dict__.items():
            if isinstance(item[1], Parameter):
                parameters.append(item[1])

        for module in self.sub_modules:
            parameters += module.parameters()

        return parameters

    def reset(self) -> None:
        """Resets temporary values like outputs and gradients."""
        self.y = None
        self.backward = None

        for module in self.sub_modules:
            module.reset()
