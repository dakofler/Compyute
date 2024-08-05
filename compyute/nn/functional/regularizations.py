"""Neural network regularization functions."""

from typing import Callable, Optional

from ...base_tensor import Tensor
from ...random.random import multinulli

__all__ = ["dropout"]


def dropout(
    x: Tensor, p: float = 0.5, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[Tensor], Tensor]]]:
    """Randomly sets tensor values to zero.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    p : float, optional
        Probability of values being set to zero. Defaults to ``0.5``.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], Tensor]], optional
        Gradient function.
    """
    dropout_map = multinulli(p, x.shape, device=x.device) / (1 - p)
    y = x * dropout_map

    if return_grad_fn:
        return y, (lambda dy: dy * dropout_map)
    return y, None
