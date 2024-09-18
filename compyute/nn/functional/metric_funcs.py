"""Neural network metric functions."""

import math

from ...tensor_ops.reducing import sum as cp_sum
from ...tensors import Tensor

__all__ = ["accuracy_score", "r2_score"]


def accuracy_score(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Computes the accuracy score given model predictions and target values.

    Parameters
    ----------
    y_pred : Tensor
        Predicted probability distribution.
    y_true : Tensor
        Target classes, must be of type ``int``.

    Returns
    -------
    Tensor
        Accuracy score.
    """
    return cp_sum(y_pred.argmax(axis=-1) == y_true) / math.prod(y_pred.shape[:-1])


def r2_score(y_pred: Tensor, y_true: Tensor, eps: float = 1e-8) -> Tensor:
    """Computes the coefficient of determination (R2 score).

    Parameters
    ----------
    y_pred : Tensor
        Model predictions.
    y_true : Tensor
        Target values.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-8``.

    Returns
    -------
    Tensor
        R2 score.
    """
    diff = y_true - y_pred
    diff_mean = y_true - y_true.mean()
    ssr = cp_sum(diff * diff)
    sst = cp_sum(diff_mean * diff_mean)
    return 1.0 - ssr / (sst + eps)
