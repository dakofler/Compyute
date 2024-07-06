"""Neural network containers module"""

from abc import abstractmethod
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

    def __init__(self, *args: Module, label: Optional[str] = None, training: bool = False) -> None:
        """Container base module.

        Parameters
        ----------
        *args : Module
            Modules used in the container.
        label: str, optional
            Container label.
        training: bool, optional
            Whether the module should be in training mode, by default False.
        """
        super().__init__(label, training)
        self._modules = list(args) if len(args) > 0 else None

    # ----------------------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------------------

    def to_device(self, device: _DeviceLike) -> None:
        device = Device(device)
        if self.device == device:
            return

        super().to_device(device)

        for module in self.modules:
            module.to_device(device)

    @property
    def modules(self) -> list[Module]:
        """Returns the list of modules."""
        if self._modules is not None:
            return [m for m in self._modules]
        return [i[1] for i in self.__dict__.items() if isinstance(i[1], Module)]

    def add(self, module: Module) -> None:
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
        if self.retain_values == value:
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
        if self.training == value:
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

    def reset(self) -> None:
        """Resets temporary values like outputs and gradients."""
        super().reset()

        for module in self.modules:
            module.reset()

    def summary(self, input_shape: _ShapeLike, input_dtype: _DtypeLike = Dtype.FLOAT32) -> None:
        """Prints information about the container and its modules.

        Parameters
        ----------
        root_module: Module
            Module to get the summary from.
        input_shape : ShapeLike
            Shape of the model input ignoring the batch dimension.
        input_dtype : DtypeLike
            Data type of the expected input data.
        """
        seperator = "=" * 75

        summary = [
            self.label,
            seperator,
            f"{'Layer':25s} {'Output Shape':20s} {'# Parameters':>15s} {'trainable':>12s}",
            seperator,
        ]

        x = ones((1,) + input_shape, dtype=input_dtype, device=self.device)
        retain_values = self.retain_values
        self.set_retain_values(True)
        _ = self(x)

        module_summaries = []

        def build_module_summary_dict(module: Module, summaries: list[dict], depth: int) -> None:
            # add summary of current modules
            s = {}
            s["name"] = " " * depth + module.label
            s["out_shape"] = (-1,) + module.y.shape[1:]
            s["n_params"] = sum(p.size for p in module.parameters)
            s["trainable"] = module.trainable
            summaries.append(s)

            # get summary of child modules
            if isinstance(module, Container):
                for module in module.modules:
                    build_module_summary_dict(module, summaries, depth + 1)

        build_module_summary_dict(self, module_summaries, 0)

        # convert dict to list of strings
        n_parameters = 0
        n_train_parameters = 0

        for s in module_summaries:
            name = s["name"]
            out_shape = str(s["out_shape"])
            n_params = s["n_params"]
            trainable = str(s["trainable"])
            n_parameters += s["n_params"]
            n_train_parameters += s["n_params"] if s["trainable"] else 0
            summary.append(f"{name:25s} {out_shape:20s} {n_params:15d} {trainable:>12s}")

        self.reset()
        self.set_retain_values(retain_values)
        summary.append(seperator)
        summary.append(f"Parameters: {n_parameters}")
        summary.append(f"Trainable parameters: {n_train_parameters}")

        summary = "\n".join(summary)
        print(summary)


class Sequential(Container):
    """Sequential container module. Layers are processed sequentially.

    Parameters
    ----------
    *args : Module
        Layers used in the sequential container.
    label: str, optional
        Container label.
    """

    def forward(self, x: Tensor) -> Tensor:
        if len(self.modules) == 0:
            raise ValueError("No modules have been added yet.")

        for module in self.modules:
            x = module(x)

        if self.training:

            def _backward(dy: Tensor) -> Tensor:
                for module in reversed(self.modules):
                    dy = module.backward(dy)
                return dy

            self._backward = _backward

        return x


class ParallelConcat(Container):
    """Parallel container module. Inputs are processed in parallel, outputs are concatinated."""

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
        if len(self.modules) == 0:
            raise ValueError("No modules have been added yet.")

        ys = [m(x) for m in self.modules]
        y = concatenate(ys, axis=self.concat_axis)

        if self.training:
            out_lens = [y.shape[self.concat_axis] for y in ys]
            splits = [sum(out_lens[: i + 1]) for i in range(len(out_lens) - 1)]

            def _backward(dy: Tensor) -> Tensor:
                dy_splits = split(dy, splits=splits, axis=self.concat_axis)
                return tensorsum(self.modules[i].backward(s) for i, s in enumerate(dy_splits))

            self._backward = _backward

        return y


class ParallelAdd(Container):
    """Parallel container module. Inputs are processed in parallel, outputs are added element-wise."""

    def forward(self, x: Tensor) -> Tensor:
        if len(self.modules) == 0:
            raise ValueError("No modules have been added yet.")

        y = tensorsum(m(x) for m in self.modules)

        if self.training:
            self._backward = lambda dy: tensorsum(m.backward(dy) for m in self.modules)

        return y
