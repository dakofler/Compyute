"""Neural network embedding functions."""

from ...preprocessing.basic import one_hot_encode
from ...tensors import Tensor
from ...typing import is_integer
from .functions import Function, FunctionCache, PseudoCache

__all__ = ["embedding"]


class EmbeddingFn(Function):
    """Performs lookup embedding on a tensor of indices."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, emb_table: Tensor) -> Tensor:
        if not is_integer(x.dtype):
            raise ValueError(f"Input must be an integer, got '{x.dtype}'.")
        y = emb_table[x]
        cache.push(x, emb_table.shape[0])
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, n_embs = cache.pop()
        batch_axes = tuple(range(x.n_axes - 1))
        x = one_hot_encode(x, n_embs, dy.dtype)
        return (x.T @ dy).sum(batch_axes)


def embedding(x: Tensor, embedding_table: Tensor) -> Tensor:
    """Performs lookup embedding on a tensor of indices.

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
    return EmbeddingFn.forward(PseudoCache(), x, embedding_table)
