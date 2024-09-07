"""Neural network base module class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from functools import wraps
from itertools import chain
from typing import Any, Optional

from ...backend import Device, DeviceError, select_device
from ...tensors import ShapeError, Tensor
from ..functional.functions import FunctionCache, PseudoCache
from ..parameter import Buffer, Parameter

__all__ = ["Module", "Identity", "ModuleList"]


class Module(ABC):
    """Neural network base module.

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    label: str
    fcache: FunctionCache
    x: Optional[Tensor] = None
    y: Optional[Tensor] = None
    _buffers: OrderedDict[str, Buffer]
    _device = select_device(None)
    _is_retaining_values = False
    _is_trainable = True
    _is_training = False
    _modules: OrderedDict[str, Module]
    _parameters: OrderedDict[str, Parameter]

    def __init__(self, label: Optional[str] = None) -> None:
        self.label = label or self.__class__.__name__
        self.fcache = PseudoCache()
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._modules = OrderedDict()

    # ----------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------

    @property
    def device(self) -> Device:
        """Device the module parametes and variables are stored on."""
        return self._device

    def to_device(self, device: Device) -> None:
        """Moves the module parameters and variables to the specified device.

        Parameters
        ----------
        device : Device
            Device to move the module parameters and variables to.
        """
        if device == self._device:
            return

        self._device = device

        for t in vars(self).values():
            if isinstance(t, Tensor):
                t.ito_device(device)

        for module in self.get_modules(recursive=False):
            module.to_device(device)

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
        for module in self.get_modules(recursive=False):
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
        for module in self.get_modules(recursive=False):
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
        for module in self.get_modules(recursive=False):
            module.is_training = value

    @property
    def n_modules(self) -> int:
        """Number of child modules."""
        return len(list(self.get_modules(recursive=False)))

    # ----------------------------------------------------------------------------------
    # CONTEXT MANAGERS
    # ----------------------------------------------------------------------------------

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

    # ----------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------

    def __repr__(self) -> str:
        attrs = [f"{a}={v}" for a, v in vars(self).items() if is_repr_attr(a, v)]
        repr_string = f"{self.label}(" + ", ".join(attrs) + ")"
        for module in self.get_modules(recursive=False):
            repr_string += "\n" + repr(module)
        return repr_string

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __bool__(self) -> bool:
        return True

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Buffer):
            self._buffers[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, ModuleList):
            for i, m in enumerate(value):
                self._modules[name + "." + str(i)] = m
        return super().__setattr__(name, value)

    # ----------------------------------------------------------------------------------
    # OTHER OPERATIONS
    # ----------------------------------------------------------------------------------

    def get_modules(self, recursive: bool = True) -> Iterator[Module]:
        """List of child modules.

        Returns
        -------
        Iterator[Module]
            Child modules.
        """
        for m in self._modules.values():
            yield m
            if recursive:
                yield from m.get_modules()

    def get_parameters(self, recursive: bool = True) -> Iterator[Parameter]:
        """Returns an Iterator of module parameters.

        Parameters
        ----------
        recursive : bool, optional
            Whether to include child modules. Defaults to ``True``.

        Returns
        -------
        Iterator[Parameter]
            Iterator of parameters.
        """
        for p in self._parameters.values():
            yield p
        if recursive:
            for m in self.get_modules():
                yield from m.get_parameters(recursive=False)

    def get_buffers(self, recursive: bool = True) -> Iterator[Buffer]:
        """Returns an Iterator of module buffers.

        Parameters
        ----------
        recursive : bool, optional
            Whether to include child modules. Defaults to ``True``.

        Returns
        -------
        Iterator[Buffer]
            Iterator of buffers.
        """
        for b in self._buffers.values():
            yield b
        if recursive:
            for m in self.get_modules():
                yield from m.get_buffers(recursive=False)

    def get_state_dict(self) -> OrderedDict:
        """Returns a state dict containing module parameters and buffers.

        Returns
        -------
        OrderedDict
            State dict containing parameters and buffers.
        """
        state_dict = OrderedDict()
        state_dict.update(self._parameters)
        state_dict.update(self._buffers)

        for k, m in self._modules.items():
            # get child module state dict
            m_state_dict = m.get_state_dict()

            # update child module state dict keys
            new_m_state_dict = OrderedDict()
            for key, value in m_state_dict.items():
                new_key = k + "." + key
                new_m_state_dict[new_key] = value

            # update state dict with child module state dict
            state_dict.update(new_m_state_dict)

        return state_dict

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        """Loads the module state from a state dict.

        Parameters
        ----------
        state_dict : OrderedDict
            State dict containing parameters and buffers.
        """
        state_dict_device = next(iter(state_dict.values())).device
        if state_dict_device != self.device:
            raise DeviceError(
                "Device mismatch."
                f"Module device: {self.device}, state dict device: {state_dict_device}"
            )

        for p, value in list(
            zip(chain(self.get_parameters(), self.get_buffers()), state_dict.values())
        ):
            p.data = value.data
            p.grad = value.grad

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the module.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """

    @abstractmethod
    def backward(self, dy: Tensor) -> Tensor:
        """Backward pass of the module.

        Parameters
        ----------
        dy : Tensor
            Output gradient tensor.

        Returns
        -------
        Tensor
            Input gradient tensor.
        """

    @staticmethod
    def register_forward(forward_method):
        """Decorator for registering a forward method to the module."""

        @wraps(forward_method)
        def wrapper(module: Module, x: Tensor) -> Tensor:
            module.fcache = FunctionCache() if module.is_training else PseudoCache()
            y = forward_method(module, x)
            if module.is_retaining_values:
                module.x = x
                module.y = y
            return y

        return wrapper

    @staticmethod
    def register_backward(backward_method):
        """Decorator for registering a backward method to the module."""

        @wraps(backward_method)
        def wrapper(module: Module, dy: Tensor) -> Tensor:
            if not module.is_training:
                raise AttributeError(f"{module.label} is not in training mode.")
            dy = dy.to_float()
            dx = backward_method(module, dy)
            if module.is_retaining_values and module.x and module.y:
                module.x.grad = dx
                module.y.grad = dy
            return dx

        return wrapper

    def clean(self, force: bool = False) -> None:
        """Removes temporary values like outputs and gradients.

        Parameters
        ----------
        force : bool, optional
            Whether to force clean and ignore ``retain_values``. Defaults to ``False``.
        """
        self.fcache.clear()

        if not self._is_retaining_values or force:
            self.x = self.y = None
            for p in self.get_parameters(recursive=False):
                p.grad = None

        for module in self.get_modules(recursive=False):
            module.clean(force)


class Identity(Module):
    """Identity module that just forwards inputs and gradients.

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x

    def backward(self, dy: Tensor) -> Tensor:
        return dy


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
            attr not in {"label"},
            not attr.startswith("_"),
            not isinstance(value, (Tensor, Module, ModuleList)),
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
