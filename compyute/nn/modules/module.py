"""Neural network base module class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from itertools import chain
from typing import Any, Callable, Iterable, Iterator, Optional

from ...base_tensor import ShapeError, Tensor
from ...engine import Device, _DeviceLike, available
from ..parameter import Buffer, Parameter

__all__ = ["Module", "Identity", "ModuleList"]


class Module(ABC):
    """Neural network base module.

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(self, label: Optional[str] = None) -> None:
        self.label = label if label is not None else self.__class__.__name__

        self.y: Optional[Tensor] = None
        self._backward: Optional[Callable[[Tensor], Tensor]] = None
        self._device: _DeviceLike = Device.CPU
        self._is_retaining_values: bool = False
        self._is_trainable: bool = True
        self._is_training: bool = False

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

        if self.y:
            self.y.ito_device(device)

        for p in chain(self.get_buffers(), self.get_parameters()):
            p.ito_device(device)

        for module in self.modules:
            module.to_device(device)

    @property
    def modules(self) -> list[Module]:
        """List of child modules.

        Returns
        -------
        list[Module]
            List of child modules.
        """
        all_modules = []
        for attribute in vars(self).values():
            if isinstance(attribute, Module):
                all_modules.append(attribute)
            if isinstance(attribute, ModuleList):
                for module in attribute:
                    all_modules.append(module)
        return all_modules

    @property
    def is_retaining_values(self) -> bool:
        """Whether the module should retain intermediate values such as outputs and gradients."""
        return self._is_retaining_values

    @is_retaining_values.setter
    def is_retaining_values(self, value: bool) -> None:
        """Set the module to retain intermediate values such as outputs and gradients.

        Parameters
        ----------
        value : bool
            Whether the module should retain intermediate values.
        """
        self._is_retaining_values = value
        for module in self.modules:
            module.is_retaining_values = value

    @property
    def is_trainable(self) -> bool:
        """Whether the module parameters are trainable."""
        return self._is_trainable

    @is_trainable.setter
    def is_trainable(self, value: bool) -> None:
        """Set module parameters to be trainable.

        Parameters
        ----------
        value : bool
            Whether the module parameters should be trainable.
        """
        self._is_trainable = value
        for module in self.modules:
            module.is_trainable = value

    @property
    def is_training(self) -> bool:
        """Whether the module is in training mode."""
        return self._is_training

    @is_training.setter
    def is_training(self, value: bool) -> None:
        """Sets module training mode.

        Parameters
        ----------
        value : bool
            Whether the module parameters should be trainable.
        """
        self._is_training = value
        for module in self.modules:
            module.is_training = value

    # ----------------------------------------------------------------------------------------------
    # CONTEXT MANAGERS
    # ----------------------------------------------------------------------------------------------

    @contextmanager
    def retain_values(self):
        """
        Context manager for setting the module to retain intermediate values
        such as outputs and gradients.
        """
        retain_values = self._is_retaining_values
        self.is_retaining_values = True
        try:
            yield
        finally:
            self.is_retaining_values = retain_values

    @contextmanager
    def train(self):
        """Context manager for putting the module into training mode."""
        training = self._is_training
        self.is_training = True
        try:
            yield
        finally:
            self.is_training = training

    # ----------------------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        attrs = [
            f"{a}={getattr(self, a)}"
            for a in vars(self)
            if is_repr_attr(a, getattr(self, a))
        ]
        repr_string = f"{self.label}(" + ", ".join(attrs) + ")"
        for module in self.modules:
            repr_string += "\n" + repr(module)
        return repr_string

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

    def __bool__(self) -> bool:
        return True

    # ----------------------------------------------------------------------------------------------
    # OTHER OPERATIONS
    # ----------------------------------------------------------------------------------------------

    def get_parameters(self, include_child_modules: bool = True) -> Iterator[Parameter]:
        """Returns an Iterator of module parameters.

        Parameters
        ----------
        include_child_modules : bool, optional
            Whether to include child modules. Defaults to ``True``.

        Returns
        -------
        Iterator[Parameter]
            Iterator of module and child module parameters.
        """
        self_parameters = (
            getattr(self, a)
            for a in vars(self)
            if isinstance(getattr(self, a), Parameter)
        )
        if include_child_modules:
            child_module_parameters = (
                p for module in self.modules for p in module.get_parameters()
            )
            return chain(self_parameters, child_module_parameters)
        return self_parameters

    def get_buffers(self, include_child_modules: bool = True) -> Iterator[Buffer]:
        """Returns an Iterator of module buffers.

        Parameters
        ----------
        include_child_modules : bool, optional
            Whether to include child modules. Defaults to ``True``.

        Returns
        -------
        Iterator[Buffer]
            Iterator of module and child module buffers.
        """
        self_buffers = (
            getattr(self, a) for a in vars(self) if isinstance(getattr(self, a), Buffer)
        )
        if include_child_modules:
            child_module_buffers = (
                b for module in self.modules for b in module.get_buffers()
            )
            return chain(self_buffers, child_module_buffers)
        else:
            return self_buffers

    def get_state_dict(self) -> OrderedDict:
        """Returns a state dict containing module parameters and buffers.

        Returns
        -------
        OrderedDict
            State dict containing parameters and buffers.
        """
        return OrderedDict(enumerate(chain(self.get_parameters(), self.get_buffers())))

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        """Loads the module state from a state dict.

        Parameters
        ----------
        state_dict : OrderedDict
            State dict containing parameters and buffers.
        """
        state_dict_device = next(iter(state_dict.values())).device
        if state_dict_device != self.device:
            raise ValueError(
                f"Device mismatch. Module device: {self.device}, state dict device: {state_dict_device}"
            )

        for p, value in list(
            zip(chain(self.get_parameters(), self.get_buffers()), state_dict.values())
        ):
            p.data = value.data
            p.grad = value.grad

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
        if not self._is_training:
            raise AttributeError(f"{self.label} is not in training mode.")

        if self._backward is None:
            raise ModelDefinitionError(
                """No backward function has been defined.
                If you are using a custom model, make sure to define a backward function and assign
                it to self._backward during the call of the forward method (see Compyute README)"""
            )

        dy = dy.to_float()
        self._set_dy(dy)

        if self._backward is not None and self._is_trainable:
            return self._backward(dy)
        return dy

    def _set_y(self, y: Tensor) -> None:
        if not self._is_retaining_values:
            return
        if self.y:
            self.y.data = y.data.copy()
        else:
            self.y = y.copy()

    def _set_dy(self, dy: Tensor) -> None:
        if self._is_retaining_values and self.y:
            self.y.grad = dy.copy()

    def clean(self, force: bool = False) -> None:
        """Removes temporary values like outputs and gradients.

        Parameters
        ----------
        force : bool, optional
            Whether to force clean and ignore ``retain_values``. Defaults to ``False``.
        """
        if self._is_retaining_values and not force:
            return
        self.y = None
        self._backward = None

        for p in chain(self.get_buffers(), self.get_parameters()):
            p.clean()

        for module in self.modules:
            module.clean(force)

    @staticmethod
    def _update_parameter_grad(
        parameter: Optional[Parameter], grad: Optional[Tensor]
    ) -> None:
        """Updates the parameter gradients."""
        if parameter and grad:
            parameter.grad += grad


class Identity(Module):
    """Identity module that just forwards inputs and gradients.

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def forward(self, x: Tensor) -> Tensor:
        self._backward = lambda dy: dy
        return x


class ModuleList(list):
    """List of modules.

    Parameters
    ----------
    modules : Iterable[Module]
        Modules to add to the list.
    """

    def __init__(self, modules: Iterable[Module]) -> None:
        super().__init__(modules)


class ModelDefinitionError(Exception):
    """Model definition error."""


def is_repr_attr(attr: str, value: Any) -> bool:
    """Checks if an attribute should be included int the class representation."""
    return all(
        [
            attr != "label",
            not attr.startswith("_"),
            not isinstance(value, Tensor),
            value is not None,
        ]
    )


def validate_input_axes(module: Module, x: Tensor, valid_n_axes: Iterable[int]) -> None:
    """Checks if the number of axes of a tensor is valid."""
    if x.n_axes in valid_n_axes:
        return
    vdims = ", ".join(str(d) for d in valid_n_axes)
    raise ShapeError(
        f"{module.label}: Invalid input dims {x.n_axes}. Can be one of: {vdims}."
    )
