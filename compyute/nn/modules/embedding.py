"""Embedding layers module"""

from typing import Optional

from ...base_tensor import Tensor
from ...dtypes import Dtype, _DtypeLike
from ...random import normal
from ...tensor_functions.creating import zeros_like
from ..functional.embeddings import lookup_embedding
from ..parameter import Parameter
from .module import Module

__all__ = ["Embedding"]


class Embedding(Module):
    """Lookup embedding layer.

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
    dtype : DtypeLike, optional
        Datatype of weights and biases, by default Dtype.FLOAT32.
    label : str, optional
        Module label.
    training : bool, optional
        Whether the module should be in training mode, by default False.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        super().__init__(label, training)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dtype = Dtype(dtype)

        # init weights
        self.w = Parameter(normal((vocab_size, embedding_dim), dtype=dtype), label="emb_w")

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [2])
        y, grad_fn = lookup_embedding(x, self.w, self._training)

        if self._training:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.as_type(self.dtype)
                dw = grad_fn(dy)
                self._update_parameter_grad(self.w, dw)
                return zeros_like(x)

            self._backward = _backward

        return y
