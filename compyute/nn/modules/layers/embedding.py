"""Embedding layers module"""

from ..module import Module
from ...parameter import Parameter
from ....preprocessing.basic import one_hot_encode
from ....random import normal
from ....tensor import Tensor
from ....types import DtypeLike


__all__ = ["Embedding"]


class Embedding(Module):
    """Layer used for token embedding."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dtype: DtypeLike = "float32",
    ) -> None:
        """Embedding layer used for token embedding.
        Input: (B, T)
            B ... batch, T ... time
        Output: (B, T, E)
            B ... batch, T ... time, E ... embedding dim

        Parameters
        ----------
        vocab_size : int
            Vocabulary size of the layer.
        embedding_dim : int
            Number of embedding dimensions of the layer.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dtype = dtype

        # init weights (Ci, Co)
        w = normal((vocab_size, embedding_dim))
        self.w = Parameter(w, dtype=dtype, label="w")

    def __repr__(self) -> str:
        name = self.__class__.__name__
        vocab_size = self.vocab_size
        embedding_dim = self.embedding_dim
        dtype = self.dtype
        return f"{name}({vocab_size=}, {embedding_dim=}, {dtype=})"

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [2])

        if x.dtype not in ("int32", "int64"):
            raise ValueError(f"Input must be int32 or int64, got {x.dtype}.")

        x = one_hot_encode(x, self.w.shape[0]).astype(self.dtype)
        y = x @ self.w

        if self.training:

            def backward(dy: Tensor) -> None:
                dy = dy.astype(self.dtype)
                self.set_dy(dy)
                self.w.grad = (x.transpose() @ dy).sum(axis=0)

            self.backward = backward

        self.set_y(y)
        return y
