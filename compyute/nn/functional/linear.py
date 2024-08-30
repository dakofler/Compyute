"""Neural network linear functions."""

from typing import Optional

from ...tensor_ops.transforming import einsum
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
        batch_dims = "uvxyz"[: x.n_axes - 1]

        # input grads
        dx = dy @ w

        # weight grads, equivalent to dy.T @ x and summing over all batch dims
        dw = einsum(f"{batch_dims}o,{batch_dims}i->oi", dy, x)

        # bias grads, equivalent to summing over all batch dims
        db = None if not b else einsum(f"{batch_dims}o->o", dy)

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
    Callable, optional
        Gradient function.

    See Also
    ----------
    :class:`compyute.nn.Linear`
    """
    return FLinear.forward(PseudoCache(), x, w, b)
