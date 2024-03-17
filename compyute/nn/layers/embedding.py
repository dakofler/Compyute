"""Embedding layers module"""

from compyute.nn.module import Module
from compyute.nn.parameter import Parameter
from compyute.preprocessing.basic import one_hot_encode
from compyute.random import uniform
from compyute.tensor import Tensor
from compyute.types import ArrayLike


__all__ = ["Embedding"]


class Embedding(Module):
    """Layer used for token embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dtype: str = "float32",
    ) -> None:
        """Embedding layer used for token embedding.
        Input: (B, T)
            B ... batch, T ... time
        Output: (B, T, Co)
            B ... batch, T ... time, C ... output channels (embedding dim)

        Parameters
        ----------
        in_channels : int
            Number of input channels (vocabulary size) of the layer.
        out_channels : int
            Number of output channels (embedding dimensions) of the layer.
        dtype: str, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype

        # init weights (Ci, Co)
        k = in_channels**-0.5
        w = uniform((in_channels, out_channels), -k, k)
        self.w = Parameter(w, dtype=dtype, label="w")

    def __repr__(self) -> str:
        name = self.__class__.__name__
        in_channels = self.in_channels
        out_channels = self.out_channels
        dtype = self.dtype
        return f"{name}({in_channels=}, {out_channels=}, {dtype=})"

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [2])
        x = one_hot_encode(x.int(), self.w.shape[0]).astype(self.dtype)
        y = x @ self.w

        if self.training:

            def backward(dy: ArrayLike) -> None:
                dy = dy.astype(self.dtype)
                self.set_dy(dy)
                self.w.grad = (x.transpose().data @ dy).sum(axis=0)

            self.backward = backward

        self.set_y(y)
        return y
