"""Loss functions."""

from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional

from ..tensors import Tensor
from .functional.losses import binary_cross_entropy, cross_entropy, mean_squared_error

__all__ = ["Loss", "BinaryCrossEntropy", "CrossEntropy", "MeanSquaredError"]


class Loss(ABC):
    """Loss base class."""

    backward: Optional[Callable[[], Tensor]] = None

    @abstractmethod
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Computes the loss given model predictions and target values.

        Parameters
        ----------
        y_pred : Tensor
            A model's predictions.
        y_true : Tensor
            Target values.

        Returns
        -------
        Tensor
            Computed loss value.
        """


class MeanSquaredError(Loss):
    r"""Computes the mean squared error loss.

    .. math::
        L = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
    """

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Computes the mean squared error loss.

        Parameters
        ----------
        y_pred : Tensor
            Model predictions.
        y_true : Tensor
            Target values.

        Returns
        -------
        Tensor
            Computed loss value.
        """
        loss, self.backward = mean_squared_error(y_pred, y_true, return_grad_fn=True)
        return loss


class CrossEntropy(Loss):
    r"""Computes the cross entropy loss.

    .. math::
        L = \frac{1}{N} \sum_{i=1}^N -\hat{y}_i \cdot \log(y_i)
    """

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Computes the cross entropy loss.

        Parameters
        ----------
        y_pred : Tensor
            Model logits.
        y_true : Tensor
            Target class labels, must be of type ``int``.

        Returns
        -------
        Tensor
            Cross entropy loss.
        """
        loss, self.backward = cross_entropy(y_pred, y_true, return_grad_fn=True)
        return loss


class BinaryCrossEntropy(Loss):
    r"""Computes the binary cross entropy loss.

    .. math::
        L = -\frac{1}{N} \sum_{i=1}^N \hat{y}_i \log(y_i) - (1 - \hat{y}_i) \log(1 - y_i)
    """

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Computes the binary cross entropy loss.

        Parameters
        ----------
        y_pred : Tensor
            Model logits.
        y_true : Tensor
            Binary target class labels, must be either ``0`` or ``1``.

        Returns
        -------
        Tensor
            Binary cross entropy loss.
        """
        loss, self.backward = binary_cross_entropy(y_pred, y_true, return_grad_fn=True)
        return loss


_LossLike = (
    Loss | Literal["binary_cross_entropy", "cross_entropy", "mean_squared_error"]
)
LOSSES = {
    "binary_cross_entropy": BinaryCrossEntropy,
    "cross_entropy": CrossEntropy,
    "mean_squared_error": MeanSquaredError,
}


def get_loss_function(loss: _LossLike) -> Loss:
    """Returns an instance of a loss function."""
    if isinstance(loss, Loss):
        return loss
    if loss not in LOSSES:
        raise ValueError(f"Unknown loss function: {loss}.")
    return LOSSES[loss]()
