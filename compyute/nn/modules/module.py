"""Neural network base module class."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from contextlib import contextmanager
from itertools import chain
from typing import Any, Callable, Iterable, Iterator, Optional

from ...base_tensor import ShapeError, Tensor
from ...engine import Device, _DeviceLike, available
from ..parameter import Buffer, Parameter

__all__ = ["Module", "save_module", "load_module"]


class Module(ABC):
    """Neural network base module.

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.
    """

    def __init__(self, label: Optional[str] = None, training: bool = False) -> None:
        self.label = label if label is not None else self.__class__.__name__
        self._training: bool = training

        self.y: Optional[Tensor] = None
        self._backward: Optional[Callable[[Tensor], Tensor]] = None
        self._device: _DeviceLike = Device.CPU
        self._retain_values: bool = False
        self._trainable: bool = True

    # ----------------------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------------------

    @property
    def device(self) -> _DeviceLike:
        """Device the module parametes and variables are stored on."""
        return self._device

    def to_device(self, device: _DeviceLike) -> None:
        """Moves the module parameters and variables to the specified device.

        Parameters
        ----------
        device : _DeviceLike
            Device to move the module parameters and variables to.
        """
        device = Device(device)
        if device == self._device:
            return

        available(device)
        self._device = device

        if self.y is not None:
            self.y = self.y.to_device(device)

        for p in chain(self.buffers, self.parameters):
            p._to_device(device)

    @property
    def trainable(self) -> bool:
        """Whether the module parameters are trainable."""
        return self._trainable

    def set_trainable(self, value: bool) -> None:
        """Set module parameters to be trainable.

        Parameters
        ----------
        value : bool
            Whether the module parameters should be trainable.
        """
        self._trainable = value

    @property
    def parameters(self) -> Iterator[Parameter]:
        """Returns an iterator of module parameters.

        Returns
        -------
        Iterator[Parameter]
            Iterator of module parameters.
        """
        return (getattr(self, a) for a in self.__dict__ if isinstance(getattr(self, a), Parameter))

    @property
    def buffers(self) -> Iterator[Buffer]:
        """Returns an iterator of module buffers.

        Returns
        -------
        Iterator[Variable]
            Iterator of module buffers.
        """
        return (getattr(self, a) for a in self.__dict__ if isinstance(getattr(self, a), Buffer))

    # ----------------------------------------------------------------------------------------------
    # CONTEXT MANAGERS
    # ----------------------------------------------------------------------------------------------

    def set_retain_values(self, value: bool) -> None:
        """Set the module to retain intermediate values such as outputs and gradients.

        Parameters
        ----------
        value : bool
            Whether the module should retain intermediate values.
        """
        self._retain_values = value

    @contextmanager
    def retain_values(self):
        """
        Context manager for setting the module to retain intermediate values
        such as outputs and gradients.
        """
        retain_values = self._retain_values
        self.set_retain_values(True)
        try:
            yield
        finally:
            self.set_retain_values(retain_values)

    def set_training(self, value: bool) -> None:
        """Set the module's training mode.

        Parameters
        ----------
        value : bool
            Whether the module should be in training mode.
        """
        self._training = value

    @contextmanager
    def training(self):
        """Context manager for putting the module into training mode."""
        training = self._training
        self.set_training(True)
        try:
            yield
        finally:
            self.set_training(training)

    # ----------------------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        attrs = [f"{a}={getattr(self, a)}" for a in self.__dict__ if _reprattr(a, getattr(self, a))]
        return f"{self.label}(" + ", ".join(attrs) + ")"

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
        y = self.forward(x)
        self._set_y(y)
        return y

    # ----------------------------------------------------------------------------------------------
    # OTHER OPERATIONS
    # ----------------------------------------------------------------------------------------------

    @abstractmethod
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

    def backward(self, dy: Tensor) -> Tensor:
        """Performs a backward pass through the module.

        Parameters
        ----------
        dy : :class:`compyute.Tensor`
            Output gradient tensor.

        Returns
        ----------
        Tensor
            Input gradient tensor.
        """
        if not self._training:
            raise AttributeError(f"{self.label} is not in training mode.")

        if self._backward is None:
            raise ModelDefinitionError(
                """No backward function has been defined.
                If you are using a custom model, make sure to define a backward function and assign
                it to self._backward during the call of the forward method (see Compyute README)"""
            )

        self._set_dy(dy)

        if self._backward is not None and self._trainable:
            return self._backward(dy)
        return dy

    def _set_y(self, y: Tensor) -> None:
        if not self._retain_values:
            return
        if self.y is None:
            self.y = y.copy()
        else:
            self.y.data = y.data.copy()

    def _set_dy(self, dy: Tensor) -> None:
        if self._retain_values and self.y is not None:
            self.y.grad = dy.copy()

    def cleanup(self, force: bool = False) -> None:
        """Resets temporary values like outputs and gradients.

        Parameters
        ----------
        force : bool, optional
            Whether to force cleanup and ignore ``retain_values``. Defaults to ``False``.
        """
        if self._retain_values and not force:
            return
        self.y = None
        self._backward = None

        for p in self.parameters:
            p.grad = None

    def _check_dims(self, x: Tensor, valid_dims: Iterable[int]) -> None:
        """Checks if the number of dimensions match the valid dimensions."""
        if x.ndim in valid_dims:
            return
        vdims = ", ".join(str(d) for d in valid_dims)
        raise ShapeError(f"{self.label}: Invalid input dims {x.ndim}. Can be one of: {vdims}.")

    @staticmethod
    def _update_parameter_grad(parameter: Optional[Parameter], grad: Optional[Tensor]) -> None:
        """Updates the parameter gradients."""
        if parameter is None or grad is None:
            return
        parameter.grad += grad


class Identity(Module):
    """Identity module that just forwards inputs and gradients.

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.
    """

    def forward(self, x: Tensor) -> Tensor:
        self._backward = lambda dy: dy
        return x


class ModelDefinitionError(Exception):
    """Model definition error."""


def save_module(module: Module, filepath: str) -> None:
    """Saves a module to a binary file.

    Parameters
    ----------
    module : Module
        Module to be saved.
    filepath : str
        Where to save the file to.
    """
    device = module.device
    module.to_device(Device.CPU)
    module.cleanup(force=True)
    with open(filepath, "wb") as file:
        pickle.dump(module, file)
    module.to_device(device)


def load_module(filepath: str) -> Module:
    """Load a module from a binary file.

    Parameters
    ----------
    filepath : str
        Filepath of the binary module file.

    Returns
    -------
    Module
        Loaded module.
    """
    with open(filepath, "rb") as file:
        obj = pickle.load(file)
    return obj


def _reprattr(a: str, v: Any) -> bool:
    return all([a != "label", not a.startswith("_"), not isinstance(v, Tensor), v is not None])
