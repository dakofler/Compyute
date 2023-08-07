"""Loss functions module"""

from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from walnut.tensor import Tensor, NumpyArray


__all__ = ["MSE", "Crossentropy"]


class Loss(ABC):
    """Loss base class."""

    def __init__(self):
        self.backward: Callable[[], NumpyArray] | None = None

    @abstractmethod
    def __call__(self, y: Tensor, t: Tensor) -> Tensor:
        ...


class MSE(Loss):
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
        dif = y - t

        def backward() -> NumpyArray:
            """Performs a backward pass."""
            return (dif * 2.0 / np.prod(y.shape).item()).data

        self.backward = backward

        return (dif**2).mean()


class Crossentropy(Loss):
    """Computes the crossentropy loss."""

    def __call__(self, y: Tensor, t: Tensor) -> Tensor:
        """Computes the crossentropy loss.

        Parameters
        ----------
        y : Tensor
            A model's predictions.
        t : Tensor
            Target values.

        Returns
        -------
        Tensor
            Crossentropy loss.
        """
        y += 1e-7

        def backward() -> NumpyArray:
            """Performs a backward pass."""
            return (-t / (y * y.shape[0])).data

        self.backward = backward

        return (-(y.log() * t).sum(axis=-1)).mean()
