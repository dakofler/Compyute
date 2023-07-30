"""Evaluation metrics module"""

from dataclasses import dataclass
from abc import ABC, abstractmethod

from walnut.tensor import Tensor


__all__ = ["Accuracy"]


@dataclass(slots=True)
class Metric(ABC):
    """Metric base class."""

    @abstractmethod
    def __call__(self, x: Tensor, y: Tensor) -> float:
        ...


@dataclass(slots=True)
class Accuracy(Metric):
    """Computes the percentage of correctly predicted classes."""

    def __call__(self, x: Tensor, y: Tensor) -> float:
        """Computes the accuracy score of a prediction compared to target values.

        Parameters
        ----------
        x : Tensor
            A model's predictions.
        y : Tensor
            Target values.

        Returns
        -------
        float
            Accuracy value.
        """

        # create tensor with ones where highest probabilities occur
        predicitons = (x / x.max(axis=1, keepdims=True) == 1.0) * 1.0

        # count number of correct samples
        num_correct_preds = (predicitons * y).sum().item()
        return num_correct_preds / predicitons.shape[0]
