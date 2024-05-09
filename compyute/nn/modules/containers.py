"""Neural network containers module"""

from typing import Optional
from .module import Module
from ..parameter import Parameter
from ...tensor_f import concatenate, ones, tensorsum
from ...tensor import Tensor
from ...types import DeviceLike, DtypeLike, ShapeLike


__all__ = ["Container", "Sequential", "ParallelConcat", "ParallelAdd"]


class Container(Module):
    """Container base module."""

    def __init__(self, modules: Optional[list[Module]] = None, label: Optional[str] = None) -> None:
        """Container base module.

        Parameters
        ----------
        modules : list[Module], optional
            List of modules used in the container.
        label: str, optional
            Module label.
        """
        super().__init__(label)
        self.__modules = modules

    # ----------------------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------------------

    def to_device(self, device: DeviceLike) -> None:
        if self.device == device:
            return

        super().to_device(device)

        for module in self.modules:
            module.to_device(device)

    @property
    def modules(self) -> list[Module]:
        """Returns the list of modules."""
        if self.__modules is not None:
            return self.__modules
        return [i[1] for i in self.__dict__.items() if isinstance(i[1], Module)]

    @property
    def parameters(self) -> list[Parameter]:
        """Returns the list of module parameters."""
        p = []
        for module in self.modules:
            p += module.parameters
        return p

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

        if self.modules is not None:
            for module in self.modules:
                rep += "\n" + module.__repr__()

        return rep

    # ----------------------------------------------------------------------------------------------
    # OTHER OPERATIONS
    # ----------------------------------------------------------------------------------------------

    def reset(self) -> None:
        """Resets temporary values like outputs and gradients."""
        super().reset()

        for module in self.modules:
            module.reset()

    def summary(self, input_shape: ShapeLike, input_dtype: DtypeLike = "float32") -> None:
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
        n = 63

        summary = [f"{self.label}\n{'-' * n}"]
        summary += [f"\n{'Layer':25s} {'Output Shape':20s} {'# Parameters':>15s}\n"]
        summary += ["=" * n, "\n"]

        x = ones((1,) + input_shape, dtype=input_dtype, device=self.device)
        retain_values = self.retain_values
        self.set_retain_values(True)
        _ = self(x)

        def build_summary(module, summary, depth):
            name = " " * depth + module.label
            output_shape = str((-1,) + module.y.shape[1:])
            n_params = sum(p.size for p in module.parameters)
            summary += [f"{name:25s} {output_shape:20s} {n_params:15d}\n"]

            if isinstance(module, Container):
                for module in module.modules:
                    build_summary(module, summary, depth + 1)

        build_summary(self, summary, 0)
        summary += ["=" * n]
        n_parameters = sum(p.size for p in self.parameters)

        self.reset()
        self.set_retain_values(retain_values)
        summary = "".join(summary)
        print(f"{summary}\n\nTotal parameters: {n_parameters}")


class Sequential(Container):
    """Sequential container module. Layers are processed sequentially."""

    def __init__(self, layers: Optional[list[Module]] = None, label: Optional[str] = None) -> None:
        """Sequential container module. Layers are processed sequentially.

        Parameters
        ----------
        layers : list[Module], optional
            List of layers used in the container.
            These layers are processed sequentially starting at index 0.
        label: str, optional
            Module label.
        """
        super().__init__(layers, label)

    def forward(self, x: Tensor) -> Tensor:
        if self.modules is None:
            raise ValueError("No modules have been added yet.")

        for module in self.modules:
            x = module(x)

        if self.training:

            def backward(dy: Tensor) -> Tensor:
                for module in reversed(self.modules):
                    dy = module.backward(dy)
                return dy

            self.backward_fn = backward

        return x


class ParallelConcat(Container):
    """Parallel container module.
    Inputs are processed independently and outputs are concatinated.
    """

    def __init__(
        self, modules: list[Module], concat_axis: int = -1, label: Optional[str] = None
    ) -> None:
        """Parallel container module. Module output tensors are concatinated.

        Parameters
        ----------
        modules : list[Module]
            List of modules used in the container.
            These modules are processed in parallel and their outputs are concatenated.
        concat_axis : int, optional
            Axis along which the output of the parallel modules
            shall be concatinated, by default -1.
        label: str, optional
            Module label.
        """
        super().__init__(modules, label)
        self.concat_axis = concat_axis

    def forward(self, x: Tensor) -> Tensor:
        if self.modules is None:
            raise ValueError("No modules have been added yet.")

        ys = [m(x) for m in self.modules]
        y = concatenate(ys, axis=self.concat_axis)

        if self.training:

            def backward(dy: Tensor) -> Tensor:
                out_lens = [y.shape[self.concat_axis] for y in ys]
                splits = [sum(out_lens[: i + 1]) for i in range(len(out_lens) - 1)]
                dy_splits = dy.split(splits, axis=self.concat_axis)
                return tensorsum([self.modules[i].backward(s) for i, s in enumerate(dy_splits)])

            self.backward_fn = backward

        return y


class ParallelAdd(Container):
    """Parallel container module.
    Inputs are processed independently and outputs are added element-wise.
    """

    def __init__(self, modules: Optional[list[Module]], label: Optional[str] = None) -> None:
        """Parallel container module. Module output tensors are added.

        Parameters
        ----------
        modules : list[Module], optional
            List of modules used in the container.
            These modules are processed in parallel and their outputs are added.
        label: str, optional
            Module label.
        """
        super().__init__(modules, label)

    def forward(self, x: Tensor) -> Tensor:
        if self.modules is None:
            raise ValueError("No modules have been added yet.")

        y = tensorsum([module(x) for module in self.modules])

        if self.training:
            self.backward_fn = lambda dy: tensorsum(
                [module.backward(dy) for module in self.modules]
            )

        return y
