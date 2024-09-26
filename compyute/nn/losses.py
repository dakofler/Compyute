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

    @staticmethod
    def register_forward(fwd_fn: Callable) -> Callable:
        """Registers a the forward method to the loss function."""

        @wraps(fwd_fn)
        def wrapper(l: Loss, y_pred: Tensor, y_true: Tensor) -> Tensor:
            l.fcache.cache.clear()

            if DEBUG:
                dt = time.perf_counter()
                y = fwd_fn(l, y_pred, y_true)
                dt = (time.perf_counter() - dt) * 1e3
                print(
                    f"{l.label:20s} | fwd | "
                    f"{y_pred.dtype:15s} | "
                    f"{y_true.dtype:15s} | "
                    f"{y.dtype:15s} | "
                    f"{dt=:>10.4f} ms"
                )
            else:
                y = fwd_fn(l, y_pred, y_true)

            return y

        return wrapper

    @staticmethod
    def register_backward(bwd_fn: Callable) -> Callable:
        """Registers a the backward method to the loss function."""

        @wraps(bwd_fn)
        def wrapper(l: Loss) -> Tensor:
            if DEBUG:
                dt = time.perf_counter()
                dx = bwd_fn(l)
                dt = (time.perf_counter() - dt) * 1e3
                print(f"{l.label:20s} | bwd | {dx.dtype:15s} | {dt=:>10.4f} ms")
            else:
                dx = bwd_fn(l)

            assert not l.fcache.cache, "FunctionCache not empty after backward."
            return dx

        return wrapper


class MeanSquaredError(Loss):
    r"""Computes the mean squared error loss.

    .. math::
        L = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
    """

    @Loss.register_forward
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return MeanSquaredErrorFn.forward(self.fcache, y_pred, y_true)

    @Loss.register_backward
    def backward(self) -> Tensor:
        return MeanSquaredErrorFn.backward(self.fcache)


class CrossEntropy(Loss):
    r"""Computes the cross entropy loss.

    .. math::
        L = \frac{1}{N} \sum_{i=1}^N -\hat{y}_i \cdot \log(y_i)
    """

    @Loss.register_forward
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return CrossEntropyFn.forward(self.fcache, y_pred, y_true)

    @Loss.register_backward
    def backward(self) -> Tensor:
        return CrossEntropyFn.backward(self.fcache)


class BinaryCrossEntropy(Loss):
    r"""Computes the binary cross entropy loss.

    .. math::
        L = -\frac{1}{N} \sum_{i=1}^N \hat{y}_i \log(y_i) - (1 - \hat{y}_i) \log(1 - y_i)
    """

    @Loss.register_forward
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return BinaryCrossEntropyFn.forward(self.fcache, y_pred, y_true)

    @Loss.register_backward
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
