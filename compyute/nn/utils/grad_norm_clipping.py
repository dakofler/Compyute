"""Gradient norm clipping utils."""

from typing import Iterator

from ...tensor_functions.creating import concatenate
from ...tensor_functions.reshaping import flatten
from ...tensor_functions.transforming import norm
from ..parameter import Parameter

__all__ = ["clip_grad_norm"]


def clip_grad_norm(parameters: Iterator[Parameter], max_norm: float) -> None:
    """Clips gradient norm of parameters.

    Parameters
    ----------
    parameters : Iterator[Parameter]
        Parameters to clip.
    max_norm : float
        Max gradient norm.
    """
    params = list(parameters)
    flattened_grads = concatenate([flatten(p.grad) for p in params if p.grad is not None])
    grad_norm = norm(flattened_grads).item()

    if grad_norm <= max_norm:
        return

    clip_coef = max_norm / grad_norm
    for p in params:
        if p.grad is not None:
            p.grad *= clip_coef
