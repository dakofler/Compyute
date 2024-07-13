"""Neural network functions module"""

from typing import Callable, Optional

from ...base_tensor import Tensor
from ...tensor_functions.computing import einsum
from ...tensor_functions.transforming import sum as cpsum

__all__ = ["linear"]


def linear(
    x: Tensor, w: Tensor, b: Optional[Tensor] = None, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], tuple[Tensor, Optional[Tensor], Optional[Tensor]]]]]:
    """Applies the linear transformation X @ W^T + b.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w : Tensor
        Weight tensor.
    b : Tensor, optional
        Bias tensor, by default None
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Linearly transformed tensor.
    Callable[[Tensor], tuple[Tensor, Tensor, Optional[Tensor]]], optional
        Gradient function.
    """
    y = x @ w.T
    if b is not None:
        y += b

    if return_grad_func:
        batch_dims = "abcdef"[: x.ndim - 1]

        def grad_func(dy: Tensor) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
            # input grads
            dx = dy @ w

            # weight grads
            if w.requires_grad:
                dw = einsum(f"{batch_dims}o,{batch_dims}i->oi", dy, x)  # sum over all batch dims
            else:
                dw = None

            # bias grads
            if b is not None and b.requires_grad:
                db = einsum(f"{batch_dims}o->o", dy)  # sum over all batch dims
            else:
                db = None

            return dx, dw, db

        return y, grad_func

    return y, None
