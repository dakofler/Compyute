"""Neural network embedding functions."""

from ...preprocessing.basic import one_hot_encode
from ...tensor_ops.transforming import sum as cp_sum
from ...tensors import Tensor
from ...typing import is_integer
from .functions import Function, FunctionCache, PseudoCache

__all__ = ["embedding"]


class FEmbedding(Function):
    """Performs lookup embedding on a tensor of indices."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, embedding_table: Tensor) -> Tensor:
        if not is_integer(x.dtype):
            raise ValueError(f"Input must be an integer, got '{x.dtype}'.")
        y = embedding_table[x]
        cache.x, cache.emb_table_shape = x, embedding_table.shape
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, emb_table_shape = cache.x, cache.emb_table_shape
        x = one_hot_encode(x, emb_table_shape[0]).to_type(dy.dtype)
        return cp_sum(x.T @ dy, axis=tuple(range(x.n_axes - 2)))


def embedding(x: Tensor, embedding_table: Tensor) -> Tensor:
    """Performs lookup embedding on a tensor of indices.

    Parameters
    ----------
    x : Tensor
        Input tensor containing indeces. Must be of type ``int8``.
    embedding_table : Tensor
        Tensor of embedding values.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return FEmbedding.forward(PseudoCache(), x, embedding_table)
