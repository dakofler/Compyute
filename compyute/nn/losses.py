"""Loss functions."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Literal

from ..tensors import Tensor
from .functional.functions import FunctionCache
from .functional.loss_funcs import (
    BinaryCrossEntropyFn,
    CrossEntropyFn,
    MeanSquaredErrorFn,
)

__all__ = ["Loss", "BinaryCrossEntropy", "CrossEntropy", "MeanSquaredError"]


DEBUG = bool(os.environ.get("COMPYUTE_DEBUG", False))


class Loss(ABC):
    """Loss base class."""

    def __init__(self) -> None:
        self.fcache = FunctionCache()
        self.label = self.__class__.__name__
        self._is_training = True

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        if not self.fcache.cache:
            self.fcache.cache.clear()

        if DEBUG:
            dt = time.perf_counter()
            y = self.forward(y_pred, y_true)
            dt = (time.perf_counter() - dt) * 1e3
            print(
                f"{self.label:20s} | forward  | {y_pred.dtype:10s} | {y_true.dtype:10s} | {dt=:>10.4f} ms"
            )
        else:
            y = self.forward(y_pred, y_true)

        return y

    def compute_grads(self) -> Tensor:
        """Computes input gradients.

        Returns
        -------
        Tensor
            Input gradients.
        """
        if DEBUG:
            dt = time.perf_counter()
            dx = self.backward()
            dt = (time.perf_counter() - dt) * 1e3
            print(f"{self.label:20s} | backward | {dx.dtype:10s} | {dt=:>10.4f} ms")
        else:
            dx = self.backward()

        assert not self.fcache.cache, "FunctionCache not empty after backward."
        return dx

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
        return MeanSquaredErrorFn.forward(self.fcache, y_pred, y_true)

    def backward(self) -> Tensor:
        return MeanSquaredErrorFn.backward(self.fcache)


class CrossEntropy(Loss):
    r"""Computes the cross entropy loss.

    .. math::
        L = \frac{1}{N} \sum_{i=1}^N -\hat{y}_i \cdot \log(y_i)
    """

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return CrossEntropyFn.forward(self.fcache, y_pred, y_true)

    def backward(self) -> Tensor:
        return CrossEntropyFn.backward(self.fcache)


class BinaryCrossEntropy(Loss):
    r"""Computes the binary cross entropy loss.

    .. math::
        L = -\frac{1}{N} \sum_{i=1}^N \hat{y}_i \log(y_i) - (1 - \hat{y}_i) \log(1 - y_i)
    """

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return BinaryCrossEntropyFn.forward(self.fcache, y_pred, y_true)

    def backward(self) -> Tensor:
        return BinaryCrossEntropyFn.backward(self.fcache)


_LossLike = (
    Loss | Literal["binary_cross_entropy", "cross_entropy", "mean_squared_error"]
)
LOSSES: dict[str, type[Loss]] = {
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
