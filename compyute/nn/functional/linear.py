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
        cache.lin_x, cache.lin_w = x, w
        y = x @ w.T

        if b:
            cache.lin_b = b
            y += b

        return y

    @staticmethod
    def backward(
        cache: FunctionCache, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        x, w, b = cache.lin_x, cache.lin_w, cache.lin_b
        batch_dims = "uvxyz"[: x.n_axes - 1]

        # input grads
        dx = dy @ w

        # weight grads, equivalent to dy.T @ x and summing over all batch dims
        dw = einsum(f"{batch_dims}o,{batch_dims}i->oi", dy, x)

        # bias grads, equivalent to summing over all batch dims
        db = einsum(f"{batch_dims}o->o", dy) if b else None

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
