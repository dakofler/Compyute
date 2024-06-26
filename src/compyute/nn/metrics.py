"""Evaluation metrics module"""

from abc import ABC, abstractmethod

from ..base_tensor import Tensor
from .functional.metrics import accuracy_score, r2_score

__all__ = ["Accuracy", "R2"]


class Metric(ABC):
    """Metric base class."""

    @abstractmethod
    def __call__(self, y: Tensor, t: Tensor) -> Tensor: ...


class Accuracy(Metric):
    """Computes the accuracy score."""

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
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
            Accuracy value.
        """
        score = accuracy_score(y_pred, y_true)
        return score


class R2(Metric):
    """Computes the coefficient of determination (R2 score)."""

    def __call__(self, y_pred: Tensor, y_true: Tensor, eps: float = 1e-8) -> Tensor:
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
        score = r2_score(y_pred, y_true, eps)
        return score


def get_metric(metric: Metric | str) -> Metric:
    """Returns an instance of a metric function."""
    if isinstance(metric, Metric):
        return metric
    metrics = {"accuracy": Accuracy, "r2": R2}
    if metric not in metrics.keys():
        raise ValueError(f"Unknown metric function {metric}.")
    return metrics[metric]()
