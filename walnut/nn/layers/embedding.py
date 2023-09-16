"""parameter layers layer"""

from __future__ import annotations

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ArrayLike
from walnut.nn.module import Module
from walnut.preprocessing.encoding import one_hot_encode


__all__ = ["Embedding"]


class Embedding(Module):
    """Layer used for token embedding."""

    def __init__(
        self, in_channels: int, out_channels: int, weights: Tensor | None = None
    ) -> None:
        """Embedding layer used for token embedding.

        Parameters
        ----------
        in_channels : int
            Number of input channels (vocabulary size) of the layer.
        out_channels : int
            Number of output channels (embedding dimensions) of the layer.
        weights : Tensor | None, optional
            Weights of the layer, by default None. If None, weights are initialized randomly.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # init weights (c_in, c_out)
        if weights is None:
            k = in_channels**-0.5
            self.w = tu.randu((in_channels, out_channels), -k, k)
        else:
            self.w = weights
        self.parameters = [self.w]

    def __repr__(self) -> str:
        name = self.__class__.__name__
        in_channels = self.in_channels
        out_channels = self.out_channels
        return f"{name}({in_channels=}, {out_channels=})"

    def __call__(self, x: Tensor) -> Tensor:
        x_enc = one_hot_encode(x, self.w.shape[0])
        y = x_enc @ self.w

        if self.training:

            def backward(dy: ArrayLike) -> None:
                self.set_dy(dy)
                self.w.grad = (x_enc.transpose((0, 2, 1)).data @ dy).sum(axis=0)

            self.backward = backward

        self.set_y(y)
        return y
