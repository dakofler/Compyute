"""Neural network loss functions."""

from functools import reduce
from operator import mul
from typing import Callable, Optional

from ...base_tensor import Tensor
from ...preprocessing.basic import one_hot_encode
from ...tensor_functions.transforming import clip, log, mean
from ...tensor_functions.transforming import sum as cpsum
from .activations import softmax

__all__ = ["mean_squared_error", "cross_entropy", "binary_cross_entropy"]


def mean_squared_error(
    y_pred: Tensor, y_true: Tensor, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[], Tensor]]]:
    """Computes the mean squared error loss.

    Parameters
    ----------
    y_pred : Tensor
        Model predictions.
    y_true : Tensor
        Target values.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Mean squared error loss.
    Callable[[], Tensor]], optional
        Gradient function.

    See Also
    --------
    :class:`compyute.nn.MeanSquaredError`
    """
    dif = y_pred.to_float() - y_true.to_float()
    loss = mean(dif**2)

    grad_fn = (lambda: dif * 2 / reduce(mul, y_pred.shape)) if return_grad_fn else None

    return loss, grad_fn


def cross_entropy(
    y_pred: Tensor, y_true: Tensor, eps: float = 1e-8, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[], Tensor]]]:
    """Computes the cross entropy loss.

    Parameters
    ----------
    y_pred : Tensor
        Model logits.
    y_true : Tensor
        Target class labels, must be of type ``int``.
    eps : float, optional
        Constant used for numerical stability. Defaults to ``1e-8``.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Cross entropy loss.
    Callable[[], Tensor]], optional
        Gradient function.

    See Also
    --------
    :class:`compyute.nn.CrossEntropy`
    """
    probs, _ = softmax(y_pred.to_float(), False)
    y_true = one_hot_encode(y_true.to_int(), y_pred.shape[-1])
    loss = -mean(log(cpsum((probs + eps) * y_true, axis=-1)))

    if return_grad_fn:

        def grad_fn() -> Tensor:
            return (probs - y_true) / reduce(mul, y_pred.shape[:-1])

        return loss, grad_fn

    return loss, None


def binary_cross_entropy(
    y_pred: Tensor, y_true: Tensor, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable[[], Tensor]]]:
    """Computes the cross entropy loss.

    Parameters
    ----------
    y_pred : Tensor
        Model logits.
    y_true : Tensor
        Binary target class labels, must be either ``0`` or ``1``.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Cross entropy loss.
    Callable[[], Tensor]], optional
        Gradient function.

    See Also
    --------
    :class:`compyute.nn.BinaryCrossEntropy`
    """
    c = 100
    loss = -mean(y_true * clip(log(y_pred), -c, c) + (1 - y_true) * clip(log(1 - y_pred), -c, c))

    if return_grad_fn:

        def grad_fn() -> Tensor:
            return (-y_true / y_pred + (1 - y_true) / (1 - y_pred)) / reduce(mul, y_pred.shape)

        return loss, grad_fn

    return loss, None
