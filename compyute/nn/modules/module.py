"""Neural network base module class."""

from __future__ import annotations

import gc
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from functools import wraps
from typing import Any, Optional

from ...backend import Device, DeviceError, free_cuda_memory, select_device
from ...tensors import ShapeError, Tensor
from ..functional.functions import FunctionCache, PseudoCache
from ..parameter import Buffer, Parameter

__all__ = ["Module", "Identity", "ModuleList"]
DEBUG = bool(os.environ.get("COMPYUTE_DEBUG", False))


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
    _retain_values = False
    _trainable = True
    _is_training = True
    _modules: OrderedDict[str, Module]
    _parameters: OrderedDict[str, Parameter]

    def __init__(self, label: Optional[str] = None) -> None:
        self.label = label or self.__class__.__name__
        self.fcache = FunctionCache()
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
    def retain_values(self) -> bool:
        """Whether the module should retain intermediate values such as outputs and gradients."""
        return self._retain_values

    @retain_values.setter
    def retain_values(self, value: bool) -> None:
        self._retain_values = value
        for module in self.get_modules(recursive=False):
            module.retain_values = value

    @property
    def trainable(self) -> bool:
        """Whether the module parameters are trainable."""
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool) -> None:
        self._trainable = value
        for module in self.get_modules(recursive=False):
            module.trainable = value

    @property
    def is_training(self) -> bool:
        """Whether the module is in training mode."""
        return self._is_training

    def training(self) -> None:
        """Puts the module in training mode."""
        self._is_training = True
        self.fcache = FunctionCache()

        for module in self.get_modules(recursive=False):
            module.training()

    def inference(self) -> None:
        """Puts the module in inference mode."""
        self._is_training = False
        self.fcache = PseudoCache()

        for module in self.get_modules(recursive=False):
            module.inference()

    @property
    def n_modules(self) -> int:
        """Number of child modules."""
        return len(list(self.get_modules(recursive=False)))

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

    def get_state_dict(self) -> OrderedDict[str, Tensor]:
        """Returns a state dict containing module parameters and buffers.

        Returns
        -------
        OrderedDict[str, Tensor]
            State dict containing parameters and buffers.
        """
        state_dict: OrderedDict[str, Tensor] = OrderedDict()
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
        for kv1, kv2 in zip(self.get_state_dict().items(), state_dict.items()):
            self_key, self_value = kv1
            other_key, other_value = kv2

            if self_key != other_key:
                raise ValueError(f"State dict key mismatch: {self_key} != {other_key}")

            if self_value.device != other_value.device:
                raise DeviceError(
                    "Device mismatch."
                    f"Module device: {self.device}, state dict device: {other_value.device}"
                )

            self_value.data = other_value.data
            self_value.grad = other_value.grad

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
            if module.retain_values:
                module.x = x

            if DEBUG:
                dt = time.time()
                y = forward_method(module, x)
                dt = (time.time() - dt) * 1e3
                print(
                    f"{module.label:20s} | forward  | {x.dtype:10s} | {y.dtype:10s} | {dt=:>10.4f} ms"
                )
            else:
                y = forward_method(module, x)

            if module.retain_values:
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

            if module.retain_values and module.y:
                module.y.grad = dy

            if DEBUG:
                dt = time.time()
                dx = backward_method(module, dy)
                dt = (time.time() - dt) * 1e3
                if dx:
                    print(
                        f"{module.label:20s} | backward | {dx.dtype:10s} | {dy.dtype:10s} | {dt=:>10.4f} ms"
                    )
                else:
                    print(
                        f"{module.label:20s} | backward | {dy.dtype:10s} | {dt=:>10.4f} ms"
                    )
            else:
                dx = backward_method(module, dy)

            assert not module.fcache, "FunctionCache not empty after backward."

            if module.retain_values and module.x:
                module.x.grad = dx
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

        if not self._retain_values or force:
            self.x = self.y = None
            for p in self.get_parameters(recursive=False):
                p.grad = None

        for module in self.get_modules(recursive=False):
            module.clean(force)

        free_cuda_memory()
        gc.collect()


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
            attr not in {"label", "fcache"},
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
