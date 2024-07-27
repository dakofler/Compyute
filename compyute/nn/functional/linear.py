"""Neural network functions module"""

from typing import Callable, Optional

from ...base_tensor import Tensor
from ...tensor_functions.computing import einsum

__all__ = ["linear"]


def linear(
    x: Tensor, w: Tensor, b: Optional[Tensor] = None, return_grad_fn: bool = False
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
    return_grad_fn : bool, optional
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

    if return_grad_fn:
        batch_dims = "abcdef"[: x.ndim - 1]

        def grad_fn(dy: Tensor) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
            # input grads
            dx = dy @ w

            # weight grads
            if w.requires_grad:
                # analogous to dy.T @ x and summing oer all batch dims
                dw = einsum(f"{batch_dims}o,{batch_dims}i->oi", dy, x)
            else:
                dw = None

            # bias grads
            if b is not None and b.requires_grad:
                # analogous to summing oer all batch dims
                db = einsum(f"{batch_dims}o->o", dy)
            else:
                db = None

            return dx, dw, db

        return y, grad_fn

    return y, None
