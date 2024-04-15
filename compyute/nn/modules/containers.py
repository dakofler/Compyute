"""Neural network containers module"""

from typing import Optional
from .module import Module
from ...tensor_f import concatenate, tensorsum
from ...tensor import Tensor


__all__ = ["SequentialContainer", "ParallelConcatContainer", "ParallelAddContainer"]


class Container(Module):
    """Container base module."""

    def __init__(self, modules: Optional[list[Module]] = None) -> None:
        """Container base module.

        Parameters
        ----------
        modules : list[Module], optional
            List of modules used in the container.
        """
        super().__init__()
        self.modules = modules

    def add(self, module: Module) -> None:
        """Adds a module.

        Parameters
        ----------
        layer : Module
            Module to append.
        """
        if self.modules is None:
            self.modules = [module]
        else:
            self.modules.append(module)


class SequentialContainer(Container):
    """Sequential container module. Layers are processed sequentially."""

    def __init__(self, layers: Optional[list[Module]] = None) -> None:
        """Sequential container module. Layers are processed sequentially.

        Parameters
        ----------
        layers : list[Module], optional
            List of layers used in the container.
            These layers are processed sequentially starting at index 0.
        """
        super().__init__(layers)

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


class ParallelConcatContainer(Container):
    """Parallel container module.
    Inputs are processed independently and outputs are concatinated.
    """

    def __init__(self, modules: list[Module], concat_axis: int = -1) -> None:
        """Parallel container module. Module output tensors are concatinated.

        Parameters
        ----------
        modules : list[Module]
            List of modules used in the container.
            These modules are processed in parallel and their outputs are concatenated.
        concat_axis : int, optional
            Axis along which the output of the parallel modules
            shall be concatinated, by default -1.
        """
        super().__init__(modules)
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
                return tensorsum(
                    [self.modules[i].backward(s) for i, s in enumerate(dy_splits)]
                )

            self.backward_fn = backward

        return y


class ParallelAddContainer(Container):
    """Parallel container module.
    Inputs are processed independently and outputs are added element-wise.
    """

    def __init__(self, modules: Optional[list[Module]]) -> None:
        """Parallel container module. Module output tensors are added.

        Parameters
        ----------
        modules : list[Module], optional
            List of modules used in the container.
            These modules are processed in parallel and their outputs are added.
        """
        super().__init__(modules)

    def forward(self, x: Tensor) -> Tensor:
        if self.modules is None:
            raise ValueError("No modules have been added yet.")

        y = tensorsum([module(x) for module in self.modules])

        if self.training:
            self.backward_fn = lambda dy: tensorsum(
                [module.backward(dy) for module in self.modules]
            )

        return y
