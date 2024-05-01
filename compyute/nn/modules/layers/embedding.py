"""Embedding layers module"""

from ..module import Module
from ...functional import lookup_embedding
from ...parameter import Parameter
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
            Vocabulary size.
        embedding_dim : int
            Embedding dimensionality.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dtype = dtype

        # init weights (Ci, Co)
        self.w = Parameter(normal((vocab_size, embedding_dim)), dtype=dtype, label="w")

    def __repr__(self) -> str:
        name = self.__class__.__name__
        vocab_size = self.vocab_size
        embedding_dim = self.embedding_dim
        dtype = self.dtype
        return f"{name}({vocab_size=}, {embedding_dim=}, {dtype=})"

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [2])
        y, emb_backward = lookup_embedding(x, self.w, self.training)

        if self.training:

            def backward(dy: Tensor) -> None:
                dy = dy.astype(self.dtype)
                self.w.grad = emb_backward(dy)

            self.backward_fn = backward

        return y
