"""loss functions module"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from walnut.tensor import Tensor

@dataclass()
class Loss(ABC):
    """Loss base class."""

    _y: np.ndarray = None
    _t: np.ndarray = None

    @abstractmethod
    def __call__(self):
        ...

    def set_vals(self, outputs: Tensor, targets: Tensor) -> None:
        """Offsets values to avoid dividing by zero."""
        self._y = outputs.data + 1e-7 # to avoid dividing by 0
        self._t = targets.data + 1e-7 # to avoid dividing by 0


class MSE(Loss):
    """Mean squard error loss function."""

    def __call__(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """Computes the mean squared error loss.

        ### Parameters:
            outputs: `Tensor`
                A model's predictions.
            targets: `Tensor`
                Target values.
        """
        super().set_vals(outputs, targets)
        return Tensor(0.5 * np.sum((self._t - self._y)**2))

    def backward(self) -> Tensor:
        """Performs a backward pass."""
        return Tensor(self._y - self._t)


class Crossentropy(Loss):
    """Crossentropy loss function."""

    def __call__(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """Computes the crossentropy loss.

        ### Parameters:
            outputs: `Tensor`
                A model's predictions.
            targets: `Tensor`
                Target values.
        """
        super().set_vals(outputs, targets)
        return Tensor(-np.mean(np.log(self._y) * self._t).item())

    def backward(self) -> Tensor:
        """Performs a backward pass."""
        return Tensor(-1.0 * self._t / (self._y * self._t.size))
