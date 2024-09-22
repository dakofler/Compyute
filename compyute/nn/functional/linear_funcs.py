"""Neural network linear functions."""

from typing import Optional

from ...tensors import Tensor
from .functions import Function, FunctionCache, PseudoCache

__all__ = ["linear"]


class LinearFn(Function):
    """Applies a linear transformation to the input."""

    @staticmethod
    def forward(
        cache: FunctionCache, x: Tensor, w: Tensor, b: Optional[Tensor]
    ) -> Tensor:
        y = x @ w.T
        if b:
            y += b
        cache.push(x, w, b is not None)
        return y

    @staticmethod
    def backward(
        cache: FunctionCache, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        x, w, b = cache.pop()
        dx = dy @ w
        dw = (dy.T @ x).sum(tuple(range(dy.n_axes - 2)))
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
    return LinearFn.forward(PseudoCache(), x, w, b)
