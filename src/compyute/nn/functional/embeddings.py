"""Neural network functions module"""

from typing import Callable, Optional

from ...base_tensor import Tensor
from ...preprocessing.basic import one_hot_encode
from ...tensor_functions.transforming import sum as _sum

__all__ = ["lookup_embedding"]


def lookup_embedding(
    x: Tensor, embedding_table: Tensor, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Optional[Tensor]]]]:
    """Performs lookup embedding on a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor of integer dtype used for indexing into the embedding table.
    embedding_table : Tensor
        Tensor of embedding values.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    if x.dtype not in ("int32", "int64"):
        raise ValueError(f"Input must be int32 or int64, got {x.dtype}.")

    x = one_hot_encode(x, embedding_table.shape[0]).astype(embedding_table.dtype)
    y = x @ embedding_table

    if return_grad_func:

        def grad_func(dy: Tensor) -> Optional[Tensor]:
            # embedding table grads
            if embedding_table.requires_grad:
                return _sum(x.T @ dy, axis=0)

        return y, grad_func

    return y, None
