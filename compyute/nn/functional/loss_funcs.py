"""Neural network loss functions."""

import math

from ...preprocessing.basic import one_hot_encode
from ...tensor_ops.unary_ops import clip, log
from ...tensors import Tensor
from .activation_funcs import softmax
from .functions import Function, FunctionCache, PseudoCache

__all__ = ["mean_squared_error", "cross_entropy", "binary_cross_entropy"]


class MeanSquaredErrorFn(Function):
    """Computes the mean squared error loss."""

    @staticmethod
    def forward(cache: FunctionCache, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred - y_true
        y = (diff * diff).mean()
        cache.push(y_pred.size, diff)
        return y

    @staticmethod
    def backward(cache: FunctionCache) -> Tensor:
        y_pred_size, diff = cache.pop()
        return 2.0 * diff / float(y_pred_size)


def mean_squared_error(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Computes the mean squared error loss.

    Parameters
    ----------
    y_pred : Tensor
        Model predictions.
    y_true : Tensor
        Target values.

    Returns
    -------
    Tensor
        Mean squared error loss.

    See Also
    --------
    :class:`compyute.nn.MeanSquaredError`
    """
    return MeanSquaredErrorFn.forward(PseudoCache(), y_pred, y_true)


class CrossEntropyFn(Function):
    """Computes the cross entropy loss."""

    @staticmethod
    def forward(
        cache: FunctionCache, y_pred: Tensor, y_true: Tensor, eta: float
    ) -> Tensor:
        probs = softmax(y_pred)
        y_true = one_hot_encode(y_true, y_pred.shape[-1], probs.dtype)
        y = (-log(probs + eta) * y_true).sum(-1).mean()
        cache.push(y_true, probs)
        return y

    @staticmethod
    def backward(cache: FunctionCache) -> Tensor:
        y_true, probs = cache.pop()
        return (probs - y_true) / float(math.prod(y_true.shape[:-1]))


def cross_entropy(y_pred: Tensor, y_true: Tensor, eta: float = 1e-8) -> Tensor:
    """Computes the cross entropy loss.

    Parameters
    ----------
    y_pred : Tensor
        Model logits.
    y_true : Tensor
        Target class labels, must be of type ``int``.
    eta : float, optional
        A small constant added for numerical stability. Defaults to ``1e-8``.


    Returns
    -------
    Tensor
        Cross entropy loss.

    See Also
    --------
    :class:`compyute.nn.CrossEntropy`
    """
    return CrossEntropyFn.forward(PseudoCache(), y_pred, y_true, eta)


class BinaryCrossEntropyFn(Function):
    """Computes the binary cross entropy loss."""

    @staticmethod
    def forward(cache: FunctionCache, y_pred: Tensor, y_true: Tensor) -> Tensor:
        log_y_pred = clip(log(y_pred), -100, 100)
        log_one_minus_y_pred = clip(log(1.0 - y_pred), -100, 100)
        y = -(y_true * log_y_pred + (1.0 - y_true) * log_one_minus_y_pred).mean()
        cache.push(y_pred, y_true)
        return y

    @staticmethod
    def backward(cache: FunctionCache) -> Tensor:
        y_pred, y_true = cache.pop()
        return (-y_true / y_pred + (1.0 - y_true) / (1.0 - y_pred)) / float(y_pred.size)


def binary_cross_entropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Computes the binary cross entropy loss.

    Parameters
    ----------
    y_pred : Tensor
        Normalized model outputs.
    y_true : Tensor
        Binary target class labels, must be either ``0`` or ``1``.

    Returns
    -------
    Tensor
        Cross entropy loss.

    See Also
    --------
    :class:`compyute.nn.BinaryCrossEntropy`
    """
    return BinaryCrossEntropyFn.forward(PseudoCache(), y_pred, y_true)
