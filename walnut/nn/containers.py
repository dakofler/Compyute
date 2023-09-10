"""Neural network containers module"""

from walnut.tensor import Tensor, ArrayLike
from walnut.nn.module import Module
import walnut.tensor_utils as tu


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

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)

                for module in reversed(self.sub_modules):
                    dy = module.backward(dy)
                return dy

            self.backward = backward

        self.set_y(x)
        return x


class ParallelContainer(Module):
    """Parallel container module."""

    def __init__(self, layers: list[Module], concat_axis: int = -1) -> None:
        """Parallel container module.

        Parameters
        ----------
        layers : list[Module]
            List of layers used in the container.
            These layers are processed in parallel and their outputs are concatenated.
        concat_axis : int, optional
            Axis along which the output of the parallel modules
            shall be concatinated, by default -1.
        """
        super().__init__()
        self.sub_modules = layers
        self.concat_axis = concat_axis

    def __call__(self, x: Tensor) -> Tensor:
        ys = [m(x) for m in self.sub_modules]
        y = tu.concatenate(ys, axis=self.concat_axis)

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                out_lens = [y.shape[self.concat_axis] for y in ys]
                splits = [sum(out_lens[: i + 1]) for i in range(len(out_lens) - 1)]
                chunks = tu.split(
                    Tensor(dy, device=self.device), splits, axis=self.concat_axis
                )

                x_grad = tu.zeros_like(x, device=self.device).data
                for i, chunk in enumerate(chunks):
                    x_grad += self.sub_modules[i].backward(chunk.data)

                return x_grad

            self.backward = backward

        self.set_y(y)
        return y
