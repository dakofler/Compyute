"""Neural network embedding functions."""

from ...preprocessing.basic import one_hot_encode
from ...tensors import Tensor
from ...typing import is_integer
from .functions import Function, FunctionCache, PseudoCache

__all__ = ["embedding"]


class EmbeddingFn(Function):
    """Performs lookup embedding on a tensor of indices."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, embed_table: Tensor) -> Tensor:
        if not is_integer(x.dtype):
            raise ValueError(f"Input must be an integer, got '{x.dtype}'.")
        y = embed_table[x]
        cache.push(x, embed_table.shape[0])
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, n_embs = cache.pop()
        batch_dims = tuple(range(x.ndim - 1))
        x = one_hot_encode(x, n_embs, dy.dtype)
        return (x.T @ dy).sum(batch_dims)


def embedding(x: Tensor, embed_table: Tensor) -> Tensor:
    """Performs lookup embedding using a tensor of integer indices.

    Parameters
    ----------
    x : Tensor
        Input tensor containing indices. Must be integers.
    emb_table : Tensor
        Tensor of embeddings.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return EmbeddingFn.forward(PseudoCache(), x, embed_table)
