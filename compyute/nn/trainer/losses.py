"""Loss functions module"""

from abc import ABC, abstractmethod
from typing import Callable, Optional
from ..functional import binary_cross_entropy, cross_entropy, mean_squared_error
from ...basetensor import Tensor


__all__ = ["BinaryCrossEntropy", "CrossEntropy", "MeanSquaredError"]


class Loss(ABC):
    """Loss base class."""

    def __init__(self):
        self.backward: Optional[Callable[[], Tensor]] = None

    @abstractmethod
    def __call__(self, y: Tensor, t: Tensor) -> Tensor: ...


class MeanSquaredError(Loss):
    """Computes the mean squared error loss."""

    def __call__(self, y: Tensor, t: Tensor) -> Tensor:
        """Computes the mean squared error loss.

        Parameters
        ----------
        y : Tensor
            A model's predictions.
        t : Tensor
            Target values.

        Returns
        -------
        Tensor
            Mean squared error loss.
        """
        loss, self.backward = mean_squared_error(y, t, return_backward_fn=True)
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

    def __call__(self, y: Tensor, t: Tensor) -> Tensor:
        """Computes the cross entropy loss.

        Parameters
        ----------
        y : Tensor
            A model's logits.
        t : Tensor
            Target class labels.

        Returns
        -------
        Tensor
            Cross entropy loss.
        """
        loss, self.backward = cross_entropy(y, t, return_backward_fn=True)
        return loss


class BinaryCrossEntropy(Loss):
    """Computes the binary cross entropy loss from model logits."""

    def __call__(self, y: Tensor, t: Tensor) -> Tensor:
        """Computes the binary cross entropy loss.

        Parameters
        ----------
        y : Tensor
            A model's logits.
        t : Tensor
            Target class labels.

        Returns
        -------
        Tensor
            Binary cross entropy loss.
        """
        loss, self.backward = binary_cross_entropy(y, t, return_backward_fn=True)
        return loss


LOSSES = {
    "binary_cross_entropy": BinaryCrossEntropy,
    "cross_entropy": CrossEntropy,
    "mean_squared_error": MeanSquaredError,
}


def get_loss(loss: Loss | str) -> Loss:
    """Returns an instance of a loss function."""
    if isinstance(loss, Loss):
        return loss
    if loss not in LOSSES.keys():
        raise ValueError(f"Unknown loss function {loss}.")
    return LOSSES[loss]()
