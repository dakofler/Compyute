"""Neural network containers module"""

from compyute.nn.module import Module
from compyute.functional import concatenate
from compyute.tensor import Tensor, ArrayLike


__all__ = ["Sequential", "ParallelConcat", "ParallelAdd"]


class Container(Module):
    """Container base module."""

    def __init__(self, modules: list[Module]) -> None:
        """Container base module.

        Parameters
        ----------
        modules : list[Module]
            List of modules used in the container.
        """
        super().__init__()
        self.child_modules = modules


class Sequential(Container):
    """Sequential container module. Layers are processed sequentially."""

    def __init__(self, layers: list[Module]) -> None:
        """Sequential container module. Layers are processed sequentially.

        Parameters
        ----------
        layers : list[Module]
            List of layers used in the container.
            These layers are processed sequentially starting at index 0.
        """
        super().__init__(layers)

    def forward(self, x: Tensor) -> Tensor:
        for module in self.child_modules:
            x = module.forward(x)

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                for module in reversed(self.child_modules):
                    dy = module.backward(dy)
                return dy

            self.backward = backward

        return x


class ParallelConcat(Container):
    """Parallel container module. Output tensors are concatinated."""

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
        ys = [m.forward(x) for m in self.child_modules]
        y = concatenate(ys, axis=self.concat_axis)

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                out_lens = [y.shape[self.concat_axis] for y in ys]
                splits = [sum(out_lens[: i + 1])
                          for i in range(len(out_lens) - 1)]
                dy_splits = Tensor(dy, device=self.device).split(
                    splits, axis=self.concat_axis)

                return sum(
                    (
                        self.child_modules[i].backward(s.data)
                        for i, s in enumerate(dy_splits)
                    )
                )

            self.backward = backward

        return y


class ParallelAdd(Container):
    """Parallel container module. Output tensors are added."""

    def __init__(self, modules: list[Module]) -> None:
        """Parallel container module. Module output tensors are added.

        Parameters
        ----------
        modules : list[Module]
            List of modules used in the container.
            These modules are processed in parallel and their outputs are added.
        """
        super().__init__(modules)

    def forward(self, x: Tensor) -> Tensor:
        y = sum([m.forward(x) for m in self.child_modules])

        if self.training:
            self.backward = lambda dy: sum(
                (m.backward(dy) for m in self.child_modules))

        return y
