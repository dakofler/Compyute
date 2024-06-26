"""Neural network module base module"""

from __future__ import annotations

import pickle
from abc import ABC
from typing import Any, Callable, Iterable, Iterator, Optional

from ...base_tensor import Tensor
from ...engine import _check_device_availability
from ...types import ShapeError, _DeviceLike
from ..parameter import Buffer, Parameter

__all__ = ["Module", "save_module", "load_module"]


class Module(ABC):
    """Module base class."""

    def __init__(self, label: Optional[str] = None) -> None:
        """Module base class."""
        self.y: Optional[Tensor] = None
        self._backward: Optional[Callable[[Tensor], Optional[Tensor]]] = None
        self.label = label if label is not None else self.__class__.__name__
        self._device: _DeviceLike = "cpu"
        self._retain_values: bool = False
        self._training: bool = False
        self._trainable: bool = True

    # ----------------------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------------------

    @property
    def device(self) -> _DeviceLike:
        """Device the module tensors are stored on."""
        return self._device

    def to_device(self, device: _DeviceLike) -> None:
        """Moves the module to the specified device."""
        if device == self._device:
            return

        _check_device_availability(device)
        self._device = device

        if self.y is not None:
            self.y.to_device(device)

        for b in self.buffers:
            b.to_device(device)

        for p in self.parameters:
            p.to_device(device)

    @property
    def retain_values(self) -> bool:
        """Whether module parameters are trainable."""
        return self._retain_values

    def set_retain_values(self, value: bool) -> None:
        """Whether module parameters are trainable."""
        self._retain_values = value

    @property
    def trainable(self) -> bool:
        """Whether the module parameters are trainable."""
        return self._trainable

    def set_trainable(self, value: bool) -> None:
        """Whether the module parameters are trainable."""
        if self._trainable == value:
            return
        self._trainable = value

        for parameter in self.parameters:
            parameter.requires_grad = value

    @property
    def training(self) -> bool:
        """Module training mode."""
        return self._training

    def set_training(self, value: bool) -> None:
        """Module training mode."""
        self._training = value

    @property
    def parameters(self) -> Iterator[Parameter]:
        """Returns module parameters."""
        return (i[1] for i in self.__dict__.items() if isinstance(i[1], Parameter))

    @property
    def buffers(self) -> Iterator[Buffer]:
        """Returns module buffers."""
        return (i[1] for i in self.__dict__.items() if isinstance(i[1], Buffer))

    # ----------------------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def _is_repr_prop(key: str, value: Any) -> bool:
        return all(
            [
                key not in ["y", "_backward", "label"],
                not key.startswith("_"),
                not isinstance(value, Tensor),
                value is not None,
            ]
        )

    def __repr__(self) -> str:
        rep = f"{self.label}("
        attributes = [f"{k}={v}" for k, v in self.__dict__.items() if self._is_repr_prop(k, v)]
        return rep + ", ".join(attributes) + ")"

    def __call__(self, x: Tensor) -> Tensor:
        y = self.forward(x)
        self._set_y(y)
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
        if not self.training:
            raise AttributeError("Model is not in training mode.")
        self._set_dy(dy)
        if self._backward is not None:
            return self._backward(dy)
        return dy

    def _set_y(self, y: Tensor) -> None:
        """Saves the module output to y tensor.

        Parameters
        ----------
        y : Tensor
            Module output tensor.
        """
        if self.retain_values:
            if self.y is None:
                self.y = y.copy()
            else:
                self.y.data = y.data.copy()

    def _set_dy(self, dy: Tensor) -> None:
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
        self._backward = None

        for p in self.parameters:
            p.grad = None

    def _check_dims(self, x: Tensor, valid_dims: Iterable[int]) -> None:
        """Checks if a tensors dimensions match desired target dimensions.

        Parameters
        ----------
        x : Tensor
            Tensor whose dimensions are checked.
        valid_dims : Iterable[int]
            Valid numbers of dimension the tensor should have.

        Raises
        ------
        ShapeError
            If the tensor's dimensions do not match the target dimensions.
        """
        if x.ndim not in valid_dims:
            vdims = ", ".join(str(d) for d in valid_dims)
            raise ShapeError(
                f"{self.label}: Number of input dimensions {x.ndim} is not valid (valid: {vdims})"
            )


def save_module(module: Module, filepath: str) -> None:
    """Saves a model as a binary file.

    Parameters
    ----------
    model : Model
        Model to be saved.
    filepath : str
        Path to the file.
    """

    module.to_device("cpu")
    module.reset()

    with open(filepath, "wb") as file:
        pickle.dump(module, file)


def load_module(filepath: str) -> Module:
    """Load a module from a previously saved binary file.

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    Model
        Loaded model.
    """
    with open(filepath, "rb") as file:
        obj = pickle.load(file)
    return obj
