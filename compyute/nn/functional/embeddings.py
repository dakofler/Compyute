"""Neural network embedding functions."""

from ...preprocessing.basic import one_hot_encode
from ...tensor_ops.transforming import einsum
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
        cache.x, cache.embedding_table = x, embedding_table
        return embedding_table[x]

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        x, embedding_table = cache.x, cache.embedding_table
        batch_dims = "uvxyz"[: dy.n_axes - 1]

        x = one_hot_encode(x, embedding_table.shape[0]).to_type(dy.dtype)
        # weight grads, equivalent to x.T @ dy and summing over all batch dims
        return einsum(f"{batch_dims}i,{batch_dims}o->io", x, dy)


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
