"""Neural network recurrent functions."""

from typing import Literal, Optional

from ...tensor_ops.creating import zeros, zeros_like
from ...tensors import Tensor
from .activations import FReLU, FTanh
from .functions import Function, FunctionCache, PseudoCache
from .linear import FLinear

__all__ = ["recurrent"]


class FRecurrent(Function):
    """Applies the recurrent function on a tensor."""

    @staticmethod
    def forward(
        cache: FunctionCache,
        x: Tensor,
        w_i: Tensor,
        b_i: Optional[Tensor],
        w_h: Tensor,
        b_h: Optional[Tensor],
        activation: Literal["relu", "tanh"] = "tanh",
    ) -> Tensor:
        act = FTanh if activation == "tanh" else FReLU

        x_h = FLinear.forward(cache, x, w_i, b_i)
        h = zeros_like(x_h)
        for t in range(x.shape[1]):
            h_h = FLinear.forward(cache, h[:, t - 1], w_h, b_h)
            h[:, t] = act.forward(cache, x_h[:, t] + h_h)

        cache.h_shape, cache.act, cache.b = h.shape, act, b_i and b_h
        return h

    @staticmethod
    def backward(
        cache: FunctionCache, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
        h_shape, act, b = cache.h_shape, cache.act, cache.b
        B, T, H = h_shape
        dpreact = zeros((B, T, H), device=dy.device, dtype=dy.dtype)
        dh = zeros((B, H), device=dy.device, dtype=dy.dtype)
        dw_h = zeros((H, H), device=dy.device, dtype=dy.dtype)
        db_h = None if not b else zeros((H,), device=dy.device, dtype=dy.dtype)

        for t in range(T - 1, -1, -1):
            dpreact[:, t] = act.backward(cache, dh + dy[:, t])
            dh, dw_h_t, db_h_t = FLinear.backward(cache, dpreact[:, t])

            if t > 0:
                dw_h += dw_h_t
            if db_h_t:
                db_h += db_h_t

        dx, dw_i, db_i = FLinear.backward(cache, dpreact)

        return dx, dw_i, db_i, dw_h, db_h


def recurrent(
    x: Tensor,
    w_i: Tensor,
    b_i: Optional[Tensor],
    w_h: Tensor,
    b_h: Optional[Tensor],
    activation: Literal["relu", "tanh"] = "tanh",
) -> Tensor:
    """Applies the recurrent function on a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w_i : Tensor
        Weight tensor for the input projection.
    b_i : Tensor, optional
        Bias tensor for the input projection. Defaults to ``None``.
    w_h : Tensor
        Weight tensor for the hidden projection.
    b_h : Tensor, optional
        Bias tensor for the hidden projection. Defaults to ``None``.
    activation : Literal["relu", "tanh"], optional
        Activation function to use. Defaults to ``tanh``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.Recurrent`
    """
    return FRecurrent.forward(PseudoCache(), x, w_i, b_i, w_h, b_h, activation)
