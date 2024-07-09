"""Neural network functions module"""

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
    y_pred: Tensor, y_true: Tensor, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[], Tensor]]]:
    """Computes the mean squared error loss.

    Parameters
    ----------
    y_pred : Tensor
        A model's predictions.
    y_true : Tensor
        Target values.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Mean squared error loss.
    Callable[[], Tensor]], optional
        Gradient function.
    """
    dif = y_pred.float() - y_true.float()
    loss = mean(dif**2)

    grad_func = (lambda: dif * 2 / reduce(mul, y_pred.shape)) if return_grad_func else None

    return loss, grad_func


def cross_entropy(
    y_pred: Tensor, y_true: Tensor, eps: float = 1e-8, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[], Tensor]]]:
    """Computes the cross entropy loss.

    Parameters
    ----------
    y_pred : Tensor
        Model logits.
    y_true : Tensor
        Target integer class labels.
    eps : float, optional
        Constant used for numerical stability, by default 1e-8.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Cross entropy loss.
    Callable[[], Tensor]], optional
        Gradient function.
    """
    probs, _ = softmax(y_pred.float(), False)
    y_true = one_hot_encode(y_true.int(), y_pred.shape[-1])
    loss = -mean(log(cpsum((probs + eps) * y_true, axis=-1)))

    if return_grad_func:

        def grad_func() -> Tensor:
            return (probs - y_true) / reduce(mul, y_pred.shape[:-1])

        return loss, grad_func

    return loss, None


def binary_cross_entropy(
    y_pred: Tensor, y_true: Tensor, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[], Tensor]]]:
    """Computes the cross entropy loss.

    Parameters
    ----------
    y_pred : Tensor
        Model logits.
    y_true : Tensor
        Binary target values.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Cross entropy loss.
    Callable[[], Tensor]], optional
        Gradient function.
    """
    c = 100
    loss = -mean(y_true * clip(log(y_pred), -c, c) + (1 - y_true) * clip(log(1 - y_pred), -c, c))

    if return_grad_func:

        def grad_func() -> Tensor:
            return (-y_true / y_pred + (1 - y_true) / (1 - y_pred)) / reduce(mul, y_pred.shape)

        return loss, grad_func

    return loss, None
