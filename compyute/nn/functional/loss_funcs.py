"""Neural network loss functions."""

import math

from ...preprocessing.basic import one_hot_encode
from ...tensor_ops.selection_ops import maximum
from ...tensor_ops.unary_ops import abs, exp, log
from ...tensors import Tensor
from .activation_funcs import sigmoid, softmax
from .functions import Function, FunctionCache, PseudoCache

__all__ = ["mean_squared_error", "cross_entropy", "binary_cross_entropy"]


class MeanSquaredErrorFn(Function):
    """Computes the mean squared error loss."""

    @staticmethod
    def forward(cache: FunctionCache, logits: Tensor, targets: Tensor) -> Tensor:
        diff = logits - targets
        loss = (diff * diff).mean()
        cache.push(logits.size, diff)
        return loss

    @staticmethod
    def backward(cache: FunctionCache) -> Tensor:
        logits_size, diff = cache.pop()
        return 2.0 * diff / float(logits_size)


def mean_squared_error(logits: Tensor, targets: Tensor) -> Tensor:
    """Computes the mean squared error loss.

    Parameters
    ----------
    logits : Tensor
        Model logits.
    targets : Tensor
        Target values.

    Returns
    -------
    Tensor
        Mean squared error loss.

    See Also
    --------
    :class:`compyute.nn.MeanSquaredError`
    """
    return MeanSquaredErrorFn.forward(PseudoCache(), logits, targets)


class CrossEntropyFn(Function):
    """Computes the cross entropy loss from logits."""

    @staticmethod
    def forward(
        cache: FunctionCache, logits: Tensor, targets: Tensor, eta: float
    ) -> Tensor:
        probs = softmax(logits)
        targets = one_hot_encode(targets, logits.shape[-1], probs.dtype)
        loss = -(log(probs + eta) * targets).sum(-1).mean()
        cache.push(targets, probs)
        return loss

    @staticmethod
    def backward(cache: FunctionCache) -> Tensor:
        targets, probs = cache.pop()
        return (probs - targets) / float(math.prod(targets.shape[:-1]))


def cross_entropy(logits: Tensor, targets: Tensor, eta: float = 1e-8) -> Tensor:
    """Computes the cross entropy loss from logits.

    Parameters
    ----------
    logits : Tensor
        Model logits.
    targets : Tensor
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
    return CrossEntropyFn.forward(PseudoCache(), logits, targets, eta)


class BinaryCrossEntropyFn(Function):
    """Computes the binary cross entropy loss from logits."""

    @staticmethod
    def forward(cache: FunctionCache, logits: Tensor, targets: Tensor) -> Tensor:
        max_logits = maximum(logits, 0.0)
        loss = (max_logits - logits * targets + log(1 + exp(-abs(logits)))).mean()
        cache.push(logits, targets)
        return loss

    @staticmethod
    def backward(cache: FunctionCache) -> Tensor:
        logits, targets = cache.pop()
        return (sigmoid(logits) - targets) / float(logits.size)  # thank you ChatGPT


def binary_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Computes the binary cross entropy loss from logits.

    Parameters
    ----------
    logits : Tensor
        Model logits.
    targets : Tensor
        Binary target class labels, must be either ``0`` or ``1``.

    Returns
    -------
    Tensor
        Binary cross entropy loss.

    See Also
    --------
    :class:`compyute.nn.BinaryCrossEntropy`
    """
    return BinaryCrossEntropyFn.forward(PseudoCache(), logits, targets)
