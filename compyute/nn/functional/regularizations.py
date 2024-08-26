"""Neural network regularization functions."""

from ...random.random import multinulli
from ...tensors import Tensor
from .functions import Function, FunctionCache, PseudoCache

__all__ = ["dropout"]


class FDropout(Function):
    """Applies the softmax function over the last axis of an input tensor."""

    @staticmethod
    def forward(
        cache: FunctionCache, x: Tensor, p: float = 0.5, training: bool = False
    ) -> Tensor:
        if not training:
            return x
        dropout_map = multinulli(p, x.shape, device=x.device) / (1 - p)
        cache.dropout_map = dropout_map
        return x * dropout_map

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor, training: bool = False) -> Tensor:
        if not training:
            return dy
        dropout_map = cache.dropout_map
        return dy * dropout_map


def dropout(x: Tensor, p: float = 0.5) -> Tensor:
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
    return FDropout.forward(PseudoCache(), x, p)
