"""Evaluation metrics module"""

from abc import ABC, abstractmethod

from walnut.tensor import Tensor
from walnut.nn.funcional import softmax


__all__ = ["Accuracy"]


class Metric(ABC):
    """Metric base class."""

    @abstractmethod
    def __call__(self, x: Tensor, y: Tensor) -> float:
        ...


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
        preds = softmax(x).argmax(-1)

        # count number of correct samples
        return (preds == y).sum().item() / y.len
