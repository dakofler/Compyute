"""Neural network regularization functions."""

from ...random.random import bernoulli
from ...tensors import Tensor
from ...typing import int8
from .functions import Function, FunctionCache, PseudoCache

__all__ = ["dropout"]


class DropoutFn(Function):
    """Randomly sets tensor values to zero."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, p: float, training: bool) -> Tensor:
        if not training or p == 0.0:
            cache.push(False, p, None)  # a bit hacky
            return x

        p = 1.0 - p
        dropout_mask = bernoulli(p, x.shape, device=x.device, dtype=int8)
        y = x * dropout_mask / p

        cache.push(True, p, dropout_mask)
        return y

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        training, p, dropout_mask = cache.pop()
        if not training:
            return dy
        return dy * dropout_mask / p


def dropout(x: Tensor, p: float = 0.5, training: bool = False) -> Tensor:
    """Randomly sets tensor values to zero.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    p : float, optional
        Probability of values being set to zero. Defaults to ``0.5``.
    training : bool, optional
        Whether to perform calculations in training mode. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return DropoutFn.forward(PseudoCache(), x, p, training)
