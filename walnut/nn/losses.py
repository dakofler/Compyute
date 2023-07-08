"""Loss functions module"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from walnut.tensor import Tensor, NumpyArray


@dataclass(init=False)
class Loss(ABC):
    """Loss base class."""

    def __init__(self) -> None:
        self.y: NumpyArray = np.empty(0, dtype="float32")
        self.t: NumpyArray = np.empty(0, dtype="float32")

    @abstractmethod
    def __call__(self, y: Tensor, t: Tensor) -> float:
        ...

    @abstractmethod
    def backward(self) -> Tensor:
        """Performs a backward pass and computes gradients."""


class MSE(Loss):
    """Mean squard error loss function."""

    def __call__(self, y: Tensor, t: Tensor) -> float:
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
        self.y = y.data + 1e-7
        self.t = t.data + 1e-7
        return np.mean(0.5 * np.sum((self.t - self.y) ** 2, axis=1)).item()

    def backward(self) -> Tensor:
        """Performs a backward pass."""
        return Tensor((self.y - self.t) / self.y.shape[0])


class Crossentropy(Loss):
    """Crossentropy loss function."""

    def __call__(self, y: Tensor, t: Tensor) -> float:
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
        self.y = y.data + 1e-7
        self.t = t.data + 1e-7
        return -np.mean(np.log(self.y) * self.t).item()

    def backward(self) -> Tensor:
        """Performs a backward pass."""
        return Tensor(-1.0 * self.t / (self.y * self.t.size))
