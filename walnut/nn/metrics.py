"""Evaluation metrics module"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from walnut import tensor
from walnut.tensor import Tensor


@dataclass
class Metric(ABC):
    """Metric base class."""

    @abstractmethod
    def __call__(self, X: Tensor, Y: Tensor) -> float:
        ...


@dataclass
class Accuracy(Metric):
    """Accuracy base class."""

    def __call__(self, X: Tensor, Y: Tensor) -> float:
        """Computes the accuracy score of a prediction compared to target values.

        Parameters
        ----------
        X : Tensor
            A model's predictions.
        Y : Tensor
            Target values.

        Returns
        -------
        float
            Accuracy value.
        """
        # create tensor with ones where highest probabilities occur
        preds = tensor.zeros_like(X).data
        p_b, _ = preds.shape
        max_prob_indices = np.argmax(X.data, axis=1)
        preds[np.arange(0, p_b), max_prob_indices] = 1

        # count number of correct samples
        num_correct_preds = np.sum(preds * Y.data).item()
        return num_correct_preds / p_b
