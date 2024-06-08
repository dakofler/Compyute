"""Embedding layers module"""

from typing import Optional

from ...._tensor import Tensor
from ...._types import _DtypeLike
from ....random import normal
from ...functional import lookup_embedding
from ...parameter import Parameter
from .._module import Module

__all__ = ["Embedding"]


class Embedding(Module):
    """Layer used for token embedding."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dtype: _DtypeLike = "float32",
        label: Optional[str] = None,
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
        label: str, optional
            Module label.
        """
        super().__init__(label)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dtype = dtype

        # init weights (Ci, Co)
        self.w = Parameter(normal((vocab_size, embedding_dim), dtype=dtype), label="w")

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [2])
        y, embedding_backward = lookup_embedding(x, self.w, self.training)

        if self.training:

            def _backward(dy: Tensor) -> None:
                dy = dy.astype(self.dtype)
                self.w.grad = embedding_backward(dy)

            self._backward = _backward

        return y
