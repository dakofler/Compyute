"""Neural network containers module"""

from walnut.tensor import Tensor, NpArrayLike
from walnut.nn.module import Module


__all__ = ["SequentialContainer", "ParallelContainer"]


class SequentialContainer(Module):
    """Sequential container module."""

    def __init__(self, layers: list[Module]) -> None:
        """Sequential container module.

        Parameters
        ----------
        layers : list[Module]
            List of layers used in the container. These layers are processed sequentially.
        """
        super().__init__()
        self.sub_modules = layers

    def __call__(self, x: Tensor) -> Tensor:
        for module in self.sub_modules:
            x = module(x)

        if self.training:

            def backward(y_grad: NpArrayLike) -> NpArrayLike:
                self.set_y_grad(y_grad)

                for module in reversed(self.sub_modules):
                    y_grad = module.backward(y_grad)
                return y_grad

            self.backward = backward

        self.set_y(x)
        return x


class ParallelContainer(Module):
    """Parallel container module."""

    def __init__(self, layers: list[Module]) -> None:
        """Parallel container module.

        Parameters
        ----------
        layers : list[Module]
            List of layers used in the container.
            These layers are processed in parallel and their outputs are concatenated.
        """
        super().__init__()
        self.sub_modules = layers

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
