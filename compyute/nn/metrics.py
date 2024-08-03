"""Evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Literal

from ..base_tensor import Tensor
from .functional.metrics import accuracy_score, r2_score

__all__ = ["Metric", "Accuracy", "R2"]


class Metric(ABC):
    """Metric base class."""

    @abstractmethod
    def __call__(self, y: Tensor, t: Tensor) -> Tensor:
        """Computes the metric score.

        Parameters
        ----------
        y_pred : Tensor
            A model's predictions.
        y_true : Tensor
            Target values.

        Returns
        -------
        Tensor
            Metric value.
        """


class Accuracy(Metric):
    """Computes the accuracy score."""

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
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

        return accuracy_score(y_pred, y_true)


class R2(Metric):
    """Computes the coefficient of determination (R2 score).

    Parameters
    ----------
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-8``.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Computes the coefficient of determination (R2 score).

        Parameters
        ----------
        y_pred : Tensor
            Model predictions.
        y_true : Tensor
            Target values.

        Returns
        -------
        Tensor
            R2 score.
        """
        return r2_score(y_pred, y_true, self.eps)


_MetricLike = Metric | Literal["accuracy", "r2"]
METRICS = {"accuracy": Accuracy, "r2": R2}


def get_metric_function(metric: _MetricLike) -> Metric:
    """Returns an instance of a metric function."""
    if isinstance(metric, Metric):
        return metric
    if metric not in METRICS:
        raise ValueError(f"Unknown metric function: {metric}.")
    return METRICS[metric]()
