"""Loss functions."""

from abc import ABC, abstractmethod
from typing import Literal

from ..tensors import Tensor
from .functional.functions import FunctionCache
from .functional.losses import FBinaryCrossEntropy, FCrossEntropy, FMeanSquaredError

__all__ = ["Loss", "BinaryCrossEntropy", "CrossEntropy", "MeanSquaredError"]


class Loss(ABC):
    """Loss base class."""

    def __init__(self) -> None:
        self.fcache = FunctionCache()

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return self.forward(y_pred, y_true)

    @abstractmethod
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
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

    @abstractmethod
    def backward(self) -> Tensor:
        """Computes the gradient of the loss with respect to the model's predictions.

        Returns
        -------
        Tensor
            The loss gradient.
        """


class MeanSquaredError(Loss):
    r"""Computes the mean squared error loss.

    .. math::
        L = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
    """

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return FMeanSquaredError.forward(self.fcache, y_pred, y_true)

    def backward(self) -> Tensor:
        return FMeanSquaredError.backward(self.fcache)


class CrossEntropy(Loss):
    r"""Computes the cross entropy loss.

    .. math::
        L = \frac{1}{N} \sum_{i=1}^N -\hat{y}_i \cdot \log(y_i)
    """

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return FCrossEntropy.forward(self.fcache, y_pred, y_true)

    def backward(self) -> Tensor:
        return FCrossEntropy.backward(self.fcache)


class BinaryCrossEntropy(Loss):
    r"""Computes the binary cross entropy loss.

    .. math::
        L = -\frac{1}{N} \sum_{i=1}^N \hat{y}_i \log(y_i) - (1 - \hat{y}_i) \log(1 - y_i)
    """

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return FBinaryCrossEntropy.forward(self.fcache, y_pred, y_true)

    def backward(self) -> Tensor:
        return FBinaryCrossEntropy.backward(self.fcache)


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
