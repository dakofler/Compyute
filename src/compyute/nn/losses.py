"""Loss functions module"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..base_tensor import Tensor
from .functional.losses import binary_cross_entropy, cross_entropy, mean_squared_error

__all__ = ["BinaryCrossEntropy", "CrossEntropy", "MeanSquaredError"]


class Loss(ABC):
    """Loss base class."""

    def __init__(self):
        self.backward: Optional[Callable[[], Tensor]] = None

    @abstractmethod
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor: ...


class MeanSquaredError(Loss):
    """Computes the mean squared error loss."""

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Computes the mean squared error loss.

        Parameters
        ----------
        y_pred : Tensor
            A model's predictions.
        y_true : Tensor
            Target values.

        Returns
        -------
        Tensor
            Mean squared error loss.
        """
        loss, self.backward = mean_squared_error(y_pred, y_true, return_grad_func=True)
        return loss


class CrossEntropy(Loss):
    """Computes the cross entropy loss from model logits."""

    def __init__(self, eps: float = 1e-8):
        """Computes the crossentropy loss from model logits.

        Parameters
        ----------
        eps : float, optional
            Constant used for numerical stability, by default 1e-8.
        """
        super().__init__()
        self.eps = eps

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Computes the cross entropy loss.

        Parameters
        ----------
        y_pred : Tensor
            A model's logits.
        y_true : Tensor
            Target class labels.

        Returns
        -------
        Tensor
            Cross entropy loss.
        """
        loss, self.backward = cross_entropy(y_pred, y_true, return_grad_func=True)
        return loss


class BinaryCrossEntropy(Loss):
    """Computes the binary cross entropy loss from model logits."""

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Computes the binary cross entropy loss.

        Parameters
        ----------
        y_pred : Tensor
            A model's logits.
        y_true : Tensor
            Target class labels.

        Returns
        -------
        Tensor
            Binary cross entropy loss.
        """
        loss, self.backward = binary_cross_entropy(y_pred, y_true, return_grad_func=True)
        return loss


def get_loss(loss: Loss | str) -> Loss:
    """Returns an instance of a loss function."""
    if isinstance(loss, Loss):
        return loss
    losses = {
        "binary_cross_entropy": BinaryCrossEntropy,
        "cross_entropy": CrossEntropy,
        "mean_squared_error": MeanSquaredError,
    }
    if loss not in losses.keys():
        raise ValueError(f"Unknown loss function {loss}.")
    return losses[loss]()
