"""Neural network regularization functions."""

from ...random.random import bernoulli
from ...tensors import Tensor
from .functions import Function, FunctionCache, PseudoCache

__all__ = ["dropout"]


class DropoutFn(Function):
    """Applies the softmax function over the last axis of an input tensor."""

    @staticmethod
    def forward(cache: FunctionCache, x: Tensor, p: float, training: bool) -> Tensor:
        if not training or p == 0.0:
            cache.push(False, None)  # a bit hacky
            return x
        dropout_map = bernoulli(1.0 - p, x.shape, device=x.device) / (1.0 - p)
        cache.push(True, dropout_map)
        return x * dropout_map

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> Tensor:
        training, dropout_map = cache.pop()
        if not training:
            return dy
        return dy * dropout_map


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
