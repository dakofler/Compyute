"""Neural network linear functions."""

from typing import Optional

from ...tensor_ops.reducing import sum as cp_sum
from ...tensors import Tensor
from .functions import Function, FunctionCache, PseudoCache

__all__ = ["linear"]


class FLinear(Function):
    """Applies a linear transformation to the input."""

    @staticmethod
    def forward(
        cache: FunctionCache, x: Tensor, w: Tensor, b: Optional[Tensor] = None
    ) -> Tensor:
        y = x @ w.T

        if b:
            y += b

        cache.x, cache.w, cache.b = x, w, b is not None
        return y

    @staticmethod
    def backward(
        cache: FunctionCache, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        x, w, b = cache.x, cache.w, cache.b

        dx = dy @ w
        dw = cp_sum(dy.T @ x, axis=tuple(range(dy.n_axes - 2)))
        db = None if not b else dy.sum(axis=tuple(range(dy.n_axes - 1)))

        return dx, dw, db


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None) -> Tensor:
    """Applies a linear transformation to the input.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w : Tensor
        Weight tensor.
    b : Tensor, optional
        Bias tensor. Defaults to ``None``. If ``None``, no bias is added.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.Linear`
    """
    return FLinear.forward(PseudoCache(), x, w, b)
