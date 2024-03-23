"""Evaluation metrics module"""

from abc import ABC, abstractmethod
from ...tensor import Tensor


__all__ = ["Accuracy", "R2"]


class Metric(ABC):
    """Metric base class."""

    @abstractmethod
    def __call__(self, y: Tensor, t: Tensor) -> Tensor: ...


class Accuracy(Metric):
    """Computes the accuracy score."""

    def __call__(self, logits: Tensor, t: Tensor) -> Tensor:
        """Computes the accuracy score.

        Parameters
        ----------
        logits : Tensor
            A model's logits.
        t : Tensor
            Target values.

        Returns
        -------
        Tensor
            Accuracy value.
        """

        return (logits.argmax(-1) == t).sum().float() / logits.shape[0]


class R2(Metric):
    """Computes the coefficient of determination (R2 score)."""

    def __call__(self, logits: Tensor, t: Tensor, eps: float = 1e-8) -> Tensor:
        """Computes the coefficient of determination (R2 score).

        Parameters
        ----------
        logits : Tensor
            A model's logits.
        t : Tensor
            Target values.
        eps: float, optional
            Constant for numerical stability, by default 1e-8.

        Returns
        -------
        Tensor
            Accuracy value.
        """

        ssr = ((t - logits) ** 2).sum()
        sst = ((t - t.mean()) ** 2).sum()
        return 1 - ssr / (sst + eps)
