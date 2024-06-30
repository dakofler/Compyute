"""Neural network functions module"""

from ...base_tensor import Tensor
from ...tensor_functions.computing import tensorprod
from ...tensor_functions.selecting import argmax
from ...tensor_functions.transforming import mean
from ...tensor_functions.transforming import sum as _sum

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
    return _sum(argmax(y_pred, -1) == y_true) / tensorprod(y_pred.shape[:-1])


def r2_score(y_pred: Tensor, y_true: Tensor, eps: float = 1e-8) -> Tensor:
    """Computes the coefficient of determination (R2 score).

    Parameters
    ----------
    y_pred : Tensor
        A model's predictions.
    y_true : Tensor
        Target values.
    eps: float, optional
        Constant for numerical stability, by default 1e-8.

    Returns
    -------
    Tensor
        R2 score.
    """
    ssr = _sum((y_true - y_pred) ** 2)
    sst = _sum((y_true - mean(y_true)) ** 2)
    return 1 - ssr / (sst + eps)
