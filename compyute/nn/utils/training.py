"""Training utilities."""

from collections.abc import Iterator

from ...tensor_ops.creation_ops import concat
from ...tensor_ops.reduction_ops import norm
from ...tensor_ops.reshape_ops import flatten
from ..parameter import Parameter

__all__ = ["clip_grad_norm"]


def clip_grad_norm(parameters: Iterator[Parameter], max_norm: float) -> float:
    """Clips gradient norm of parameters to a specified value.

    Parameters
    ----------
    parameters : Iterator[Parameter]
        Parameters to clip.
    max_norm : float
        Max gradient norm.

    Returns
    ----------
    float
        Unclipped gradient norm.
    """
    params = list(parameters)
    grads = concat([flatten(p.grad) for p in params if p.grad])
    grad_norm = norm(grads).item()

    if grad_norm <= max_norm:
        return grad_norm

    clip_coef = max_norm / grad_norm
    for p in params:
        if p.grad:
            p.grad *= clip_coef

    return grad_norm
