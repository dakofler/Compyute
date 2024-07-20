"""Neural network containers module"""

from abc import abstractmethod
from itertools import accumulate
from typing import Iterator, Optional

from ...base_tensor import Tensor, _ShapeLike
from ...dtypes import Dtype, _DtypeLike
from ...engine import Device, _DeviceLike
from ...tensor_functions.combining import concatenate, split
from ...tensor_functions.computing import tensorsum
from ...tensor_functions.creating import ones
from ..parameter import Parameter
from .module import Module

__all__ = ["Container", "Sequential", "ParallelConcat", "ParallelAdd"]


class Container(Module):
    """Container base module."""

    __slots__ = ("_modules",)

    def __init__(
        self, *modules: Module, label: Optional[str] = None, training: bool = False
    ) -> None:
        """Container base module.

        Parameters
        ----------
        *modules : Module
            Modules used in the container.
        label: str, optional
            Container label.
        training: bool, optional
            Whether the module should be in training mode, by default False.
        """
        super().__init__(label, training)
        self._modules = list(modules) if len(modules) > 0 else None

    # ----------------------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------------------

    def to_device(self, device: _DeviceLike) -> None:
        if self.device == Device(device):
            return
        super().to_device(device)
        for module in self.modules:
            module.to_device(device)

    @property
    def modules(self) -> list[Module]:
        """Returns the list of modules."""
        if self._modules is not None:
            return self._modules
        if "__dict__" in dir(self):
            return [getattr(self, a) for a in self.__dict__ if isinstance(getattr(self, a), Module)]
        return [getattr(self, a) for a in self.__slots__ if isinstance(getattr(self, a), Module)]

    def add_module(self, module: Module) -> None:
        """Adds a module to the container.

        Parameters
        ----------
        module : Module
            Module to add to the container.
        """
        if self._modules is None:
            self._modules = [module]
        else:
            self._modules.append(module)

    @property
    def parameters(self) -> Iterator[Parameter]:
        """Returns a generator of module parameters."""
        return (p for module in self.modules for p in module.parameters)

    def set_retain_values(self, value: bool) -> None:
        if self._retain_values == value:
            return
        super().set_retain_values(value)
        for module in self.modules:
            module.set_retain_values(value)

    def set_trainable(self, value: bool) -> None:
        if self.trainable == value:
            return
        super().set_trainable(value)
        for module in self.modules:
            module.set_trainable(value)

    def set_training(self, value: bool) -> None:
        if self._training == value:
            return
        super().set_training(value)
        for module in self.modules:
            module.set_training(value)

    # ----------------------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        rep = super().__repr__()

        for module in self.modules:
            rep += "\n" + module.__repr__()

        return rep

    # ----------------------------------------------------------------------------------------------
    # OTHER OPERATIONS
    # ----------------------------------------------------------------------------------------------

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...

    def cleanup(self, force: bool = False) -> None:
        """Resets temporary values like outputs and gradients."""
        super().cleanup(force)

        for module in self.modules:
            module.cleanup(force)

    def get_summary(self, input_shape: _ShapeLike, input_dtype: _DtypeLike = Dtype.FLOAT32) -> str:
        """Returns information about the container and its modules.

        Parameters
        ----------
        root_module: Module
            Module to get the summary from.
        input_shape : _ShapeLike
            Shape of the model input ignoring the batch dimension.
        input_dtype : _DtypeLike
            Data type of the expected input data.

        Returns
        -------
        str
            Summary of the container and its modules.
        """
        divider = "=" * 80

        summary = [
            self.label,
            divider,
            f"{'Layer':30s} {'Output Shape':20s} {'# Parameters':>15s} {'trainable':>12s}",
            divider,
        ]

        x = ones((1,) + input_shape, dtype=input_dtype, device=self.device)
        with self.retain_values():
            _ = self(x)
            module_summaries = []

            def build_module_summary(module: Module, summaries: list[dict], prefix: str) -> None:
                # add summary of current module
                summaries.append(
                    {
                        "name": prefix + module.label,
                        "out_shape": (-1,) + module.y.shape[1:] if module.y is not None else (),
                        "n_params": sum(p.size for p in module.parameters),
                        "trainable": module.trainable,
                        "type": "container" if isinstance(module, Container) else "module",
                    }
                )

                # get summary of child modules
                if isinstance(module, Container):
                    for i, m in enumerate(module.modules):
                        child_prefix = prefix[:-2]

                        if prefix[-2:] == "├-":
                            child_prefix += "│ "
                        elif prefix[-2:] == "└-":
                            child_prefix += "  "

                        child_prefix += "└-" if i == len(module.modules) - 1 else "├-"

                        build_module_summary(m, summaries, child_prefix)

            build_module_summary(self, module_summaries, "")

            # convert dict to list of strings
            n_parameters = 0
            n_train_parameters = 0

            for s in module_summaries:
                name = s["name"]
                out_shape = str(s["out_shape"])
                n_params = s["n_params"]
                trainable = str(s["trainable"])
                n_parameters += s["n_params"] if s["type"] == "module" else 0
                n_train_parameters += (
                    s["n_params"] if s["trainable"] and s["type"] == "module" else 0
                )
                summary.append(f"{name:30s} {out_shape:20s} {n_params:15d} {trainable:>12s}")

        self.cleanup()
        summary.append(divider)
        summary.append(f"Parameters: {n_parameters}")
        summary.append(f"Trainable parameters: {n_train_parameters}")

        return "\n".join(summary)


class Sequential(Container):
    """Sequential container module. Layers are processed sequentially."""

    __slots__ = ()

    def forward(self, x: Tensor) -> Tensor:
        if not self.modules:
            raise EmptyContainerError()

        for module in self.modules:
            x = module(x)

        if self._training:

            def _backward(dy: Tensor) -> Tensor:
                for module in reversed(self.modules):
                    dy = module.backward(dy)
                return dy

            self._backward = _backward

        return x


class ParallelConcat(Container):
    """Parallel container module. Inputs are processed in parallel, outputs are concatinated."""

    __slots__ = ("concat_axis",)

    def __init__(
        self,
        *args: Module,
        concat_axis: int = -1,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        """Parallel container module. Module output tensors are concatinated.

        Parameters
        ----------
        *args : Module
            Modules used in the parallel container.
        concat_axis : int, optional
            Axis along which the output of the parallel modules
            shall be concatinated, by default -1.
        label: str, optional
            Container label.
        training: bool, optional
            Whether the module should be in training mode, by default False.
        """
        super().__init__(*args, label=label, training=training)
        self.concat_axis = concat_axis

    def forward(self, x: Tensor) -> Tensor:
        if not self.modules:
            raise EmptyContainerError()

        ys = [m(x) for m in self.modules]
        y = concatenate(ys, axis=self.concat_axis)

        if self._training:
            splits = list(accumulate(y.shape[self.concat_axis] for y in ys[:-1]))

            def _backward(dy: Tensor) -> Tensor:
                dy_splits = split(dy, splits=splits, axis=self.concat_axis)
                return tensorsum(m.backward(s) for m, s in zip(self.modules, dy_splits))

            self._backward = _backward

        return y


class ParallelAdd(Container):
    """Parallel container module.
    Inputs are processed in parallel, outputs are added element-wise."""

    __slots__ = ()

    def forward(self, x: Tensor) -> Tensor:
        if not self.modules:
            raise EmptyContainerError()

        y = tensorsum(m(x) for m in self.modules)

        if self._training:
            self._backward = lambda dy: tensorsum(m.backward(dy) for m in self.modules)

        return y


class EmptyContainerError(Exception):
    """Exception for empty containers."""

    def __init__(self, message: str = "Container has no modules.") -> None:
        super().__init__(message)
