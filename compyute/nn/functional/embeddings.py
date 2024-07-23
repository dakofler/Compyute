"""Neural network functions module"""

from typing import Callable, Optional

from ...base_tensor import Tensor
from ...dtypes import Dtype
from ...preprocessing.basic import one_hot_encode
from ...tensor_functions.transforming import sum as cpsum

__all__ = ["lookup_embedding"]


def lookup_embedding(
    x: Tensor, embedding_table: Tensor, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Optional[Tensor]]]]:
    """Performs lookup embedding on a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor of integer dtype used for indexing into the embedding table.
    embedding_table : Tensor
        Tensor of embedding values.
    return_grad_fn: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    if x.dtype not in (Dtype.INT8, Dtype.INT16, Dtype.INT32, Dtype.INT64):
        raise ValueError(f"Input must be an integer, got '{x.dtype}'.")

    x = one_hot_encode(x, embedding_table.shape[0]).as_type(embedding_table.dtype)
    y = x @ embedding_table

    if return_grad_fn:

        def grad_fn(dy: Tensor) -> Optional[Tensor]:
            # embedding table grads
            if embedding_table.requires_grad:
                return cpsum(x.T @ dy, axis=0)

        return y, grad_fn

    return y, None
