"""Neural network embedding functions."""

from typing import Callable, Optional

from ...preprocessing.basic import one_hot_encode
from ...tensor_ops.transforming import einsum
from ...tensors import Tensor
from ...typing import is_integer

__all__ = ["lookup_embedding"]


def lookup_embedding(
    x: Tensor, embedding_table: Tensor, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable]]:
    """Performs lookup embedding on a tensor of indices.

    Parameters
    ----------
    x : Tensor
        Input tensor containing indeces. Must be of type ``int8``.
    embedding_table : Tensor
        Tensor of embedding values.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable, optional
        Gradient function.
    """
    if not is_integer(x.dtype):
        raise ValueError(f"Input must be an integer, got '{x.dtype}'.")

    if not return_grad_fn:
        return embedding_table[x], None

    x = one_hot_encode(x, embedding_table.shape[0]).to_type(embedding_table.dtype)
    y = x @ embedding_table
    batch_dims = "uvxyz"[: x.n_axes - 1]

    def grad_fn(dy: Tensor) -> Tensor:
        # weight grads, equivalent to x.T @ dy and summing over all batch dims
        return einsum(f"{batch_dims}i,{batch_dims}o->io", x, dy)

    return y, grad_fn
