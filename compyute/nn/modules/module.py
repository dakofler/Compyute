"""Neural network base module class."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from contextlib import contextmanager
from itertools import chain
from typing import Any, Callable, Iterable, Iterator, Optional

from ...base_tensor import ShapeError, Tensor, _ShapeLike
from ...dtypes import Dtype, _DtypeLike
from ...engine import Device, _DeviceLike, available
from ...tensor_ops.creating import ones
from ..parameter import Buffer, Parameter

__all__ = ["Module", "Identity"]


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
        self._modules: Optional[list[Module]] = None
        self._retain_values: bool = False
        self._trainable: bool = True
        self._training: bool = False

    # ----------------------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------------------

    @property
    def self_parameters(self) -> Iterator[Parameter]:
        """Iterator of module parameters."""
        return (getattr(self, a) for a in self.__dict__ if isinstance(getattr(self, a), Parameter))

    @property
    def parameters(self) -> Iterator[Parameter]:
        """Iterator of module and child module parameters."""
        self_parameters = self.self_parameters
        child_module_parameters = (p for module in self.modules for p in module.parameters)
        return chain(self_parameters, child_module_parameters)

    @property
    def buffers(self) -> Iterator[Buffer]:
        """Iterator of module and child module buffers."""
        self_buffers = (getattr(self, a) for a in self.__dict__ if isinstance(getattr(self, a), Buffer))
        child_module_buffers = (b for module in self.modules for b in module.buffers)
        return chain(self_buffers, child_module_buffers)

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
            self.y._to_device(device)

        for p in chain(self.buffers, self.parameters):
            p._to_device(device)

        for module in self.modules:
            module.to_device(device)

    @property
    def modules(self) -> list[Module]:
        """Returns the list of child modules.

        Returns
        -------
        list[Module]
            List of child modules.
        """
        if self._modules is not None:
            return self._modules
        return [getattr(self, a) for a in self.__dict__ if isinstance(getattr(self, a), Module)]

    @modules.setter
    def modules(self, value: list[Module]) -> None:
        """Set the list of child modules.

        Parameters
        ----------
        value : list[Module]
            List of child modules.
        """
        self._modules = value

    @property
    def retain_values(self) -> bool:
        """Whether the module should retain intermediate values such as outputs and gradients."""
        return self._retain_values

    @retain_values.setter
    def retain_values(self, value: bool) -> None:
        """Set the module to retain intermediate values such as outputs and gradients.

        Parameters
        ----------
        value : bool
            Whether the module should retain intermediate values.
        """
        self._retain_values = value
        for module in self.modules:
            module.retain_values = value

    @property
    def trainable(self) -> bool:
        """Whether the module parameters are trainable."""
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool) -> None:
        """Set module parameters to be trainable.

        Parameters
        ----------
        value : bool
            Whether the module parameters should be trainable.
        """
        self._trainable = value
        for module in self.modules:
            module.trainable = value

    @property
    def training(self) -> bool:
        """Whether the module is in training mode."""
        return self._training

    @training.setter
    def training(self, value: bool) -> None:
        """Set module parameters to be trainable.

        Parameters
        ----------
        value : bool
            Whether the module parameters should be trainable.
        """
        self._training = value
        for module in self.modules:
            module.training = value

    # ----------------------------------------------------------------------------------------------
    # CONTEXT MANAGERS
    # ----------------------------------------------------------------------------------------------

    @contextmanager
    def do_retain_values(self):
        """
        Context manager for setting the module to retain intermediate values
        such as outputs and gradients.
        """
        retain_values = self._retain_values
        self.retain_values = True
        try:
            yield
        finally:
            self.retain_values = retain_values

    @contextmanager
    def do_training(self):
        """Context manager for putting the module into training mode."""
        training = self._training
        self.training = True
        try:
            yield
        finally:
            self.training = training

    # ----------------------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        attrs = [f"{a}={getattr(self, a)}" for a in self.__dict__ if _reprattr(a, getattr(self, a))]
        repr_string = f"{self.label}(" + ", ".join(attrs) + ")"
        for module in self.modules:
            repr_string += "\n" + module.__repr__()
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

        dy = dy.to_float()
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

        for p in chain(self.buffers, self.parameters):
            p._cleanup()

        for module in self.modules:
            module.cleanup(force)

    def _check_dims(self, x: Tensor, valid_dims: Iterable[int]) -> None:
        """Checks if the number of dimensions match the valid dimensions."""
        if x.n_axes in valid_dims:
            return
        vdims = ", ".join(str(d) for d in valid_dims)
        raise ShapeError(f"{self.label}: Invalid input dims {x.n_axes}. Can be one of: {vdims}.")

    @staticmethod
    def _update_parameter_grad(parameter: Optional[Parameter], grad: Optional[Tensor]) -> None:
        """Updates the parameter gradients."""
        if parameter is None or grad is None:
            return
        parameter.grad += grad

    def save(self, filepath: str) -> None:
        """Saves the module to a binary file.

        Parameters
        ----------
        module : Module
            Module to be saved.
        filepath : str
            Where to save the file to.
        """
        device = self.device
        self.to_device(Device.CPU)
        self.cleanup(force=True)
        with open(filepath, "wb") as file:
            pickle.dump(self, file)
        self.to_device(device)

    @classmethod
    def load(cls, filepath: str) -> Module:
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
            module = pickle.load(file)
        return module

    def get_summary(self, input_shape: _ShapeLike, input_dtype: _DtypeLike = Dtype.FLOAT32) -> str:
        """Returns information about the module and its child modules.

        Parameters
        ----------
        input_shape : _ShapeLike
            Shape of the container input ignoring the batch dimension.
        input_dtype : _DtypeLike, optional
            Data type of the expected input data. Defaults to :class:`compyute.float32`.

        Returns
        -------
        str
            Summary of the module and its child modules.
        """

        def get_module_summary(module: Module, prefix: str) -> None:
            # add summary of current module
            module_summaries.append(
                {
                    "name": prefix + module.label,
                    "out_shape": (-1,) + module.y.shape[1:] if module.y is not None else (),
                    "n_params": {p.ptr: p.size for p in module.self_parameters},
                    "trainable": module.trainable,
                }
            )

            # get summary of child modules
            for i, child_module in enumerate(module.modules):
                child_prefix = prefix[:-2]
                if prefix[-2:] == "├-":
                    child_prefix += "│ "
                elif prefix[-2:] == "└-":
                    child_prefix += "  "
                child_prefix += "└-" if i == len(module.modules) - 1 else "├-"
                get_module_summary(child_module, child_prefix)

        # perform forward pass to get output shapes
        x = ones((1,) + input_shape, dtype=input_dtype, device=self.device)
        with self.do_retain_values():
            _ = self(x)

        # get model summary
        module_summaries = []
        get_module_summary(self, "")
        self.cleanup()

        # format summary
        divider = "=" * 80
        summary = [
            self.label,
            divider,
            f"{'Layer':30s} {'Output Shape':20s} {'# Parameters':>15s} {'trainable':>12s}",
            divider,
        ]

        n_params = n_train_params = 0
        param_ptrs = []

        for m in module_summaries:
            m_name = m["name"]
            m_out_shape = str(m["out_shape"])
            m_n_params = sum(m["n_params"].values())
            m_trainable = str(m["trainable"])
            summary.append(f"{m_name:30s} {m_out_shape:20s} {m_n_params:15d} {m_trainable:>12s}")

            # count parameters without duplicates (can occur with weight sharing of modules)
            for ptr, n in m["n_params"].items():
                if ptr in param_ptrs:
                    continue
                param_ptrs.append(ptr)
                n_params += n
                n_train_params += n if m["trainable"] else 0

        summary.append(divider)
        summary.append(f"Parameters: {n_params}")
        summary.append(f"Trainable parameters: {n_train_params}")

        return "\n".join(summary)


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


def _reprattr(a: str, v: Any) -> bool:
    return all([a != "label", not a.startswith("_"), not isinstance(v, Tensor), v is not None])
