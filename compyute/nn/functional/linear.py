"""Neural network functions module"""

from typing import Callable, Optional

from ...base_tensor import Tensor
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

        def grad_func(dy: Tensor) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
            # input grads
            dx = dy @ w

            # weight grads
            if w.requires_grad:
                dw = dy.T @ x
                if x.ndim > 2:  # sum over all batch dimensions
                    axes = tuple(range(x.ndim - 2))
                    dw = cpsum(dw, axis=axes)
            else:
                dw = None

            # bias grads
            if b is not None and b.requires_grad:
                axes = tuple(range(x.ndim - 1))
                db = cpsum(dy, axis=axes)  # sum over all batch dimensions
            else:
                db = None

            return dx, dw, db

        return y, grad_func

    return y, None
