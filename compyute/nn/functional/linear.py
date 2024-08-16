"""Neural network linear functions."""

from typing import Callable, Optional

from ...base_tensor import Tensor
from ...tensor_ops.transforming import einsum

__all__ = ["linear"]


def linear(
    x: Tensor, w: Tensor, b: Optional[Tensor] = None, return_grad_fn: bool = False
) -> tuple[
    Tensor, Optional[Callable[[Tensor], tuple[Tensor, Tensor, Optional[Tensor]]]]
]:
    """Applies a linear transformation to the input.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w : Tensor
        Weight tensor.
    b : Tensor, optional
        Bias tensor. Defaults to ``None``. If ``None``, no bias is added.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], tuple[Tensor, Tensor, Optional[Tensor]]], optional
        Gradient function.

    See Also
    ----------
    :class:`compyute.nn.Linear`
    """
    y = x @ w.T
    if b:
        y += b

    if return_grad_fn:
        batch_dims = "uvxyz"[: x.n_axes - 1]

        def grad_fn(dy: Tensor) -> tuple[Tensor, Tensor, Optional[Tensor]]:
            # input grads
            dx = dy @ w

            # weight grads, analogous to dy.T @ x and summing over all batch dims
            dw = einsum(f"{batch_dims}o,{batch_dims}i->oi", dy, x)

            # bias grads, analogous to summing over all batch dims
            db = einsum(f"{batch_dims}o->o", dy) if b else None

            return dx, dw, db

        return y, grad_fn

    return y, None
