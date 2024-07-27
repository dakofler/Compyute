"""Neural network metric functions."""

from functools import reduce
from operator import mul

from ...base_tensor import Tensor
from ...tensor_functions.selecting import argmax
from ...tensor_functions.transforming import mean
from ...tensor_functions.transforming import sum as cpsum

__all__ = ["accuracy_score", "r2_score"]


def accuracy_score(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Computes the accuracy score.

    Parameters
    ----------
    y_pred : Tensor
        A model's predictions.
    y_true : Tensor
        Target values.

    Returns
    -------
    Tensor
        Accuracy score.
    """
    return cpsum(argmax(y_pred, -1) == y_true) / reduce(mul, y_pred.shape[:-1])


def r2_score(y_pred: Tensor, y_true: Tensor, eps: float = 1e-8) -> Tensor:
    """Computes the coefficient of determination (R2 score).

    Parameters
    ----------
    y_pred : Tensor
        A model's predictions.
    y_true : Tensor
        Target values.
    eps : float, optional
        Constant for numerical stability, by default 1e-8.

    Returns
    -------
    Tensor
        R2 score.
    """
    ssr = cpsum((y_true - y_pred) ** 2)
    sst = cpsum((y_true - mean(y_true)) ** 2)
    return 1 - ssr / (sst + eps)
