"""loss functions module"""

import numpy as np


class Loss():
    """Loss base class."""

    def __init__(self) -> None:
        self.loss = 0
        self._y = None
        self._t = None

    def set_vals(self, outputs, targets) -> None:
        """Offsets values to avoid dividing by zero."""
        self._y = outputs + 1e-7 # to avoid dividing by 0
        self._t = targets + 1e-7 # to avoid dividing by 0


class MSE(Loss):
    """Mean squard error loss function"""

    def __call__(self, outputs, targets) -> np.ndarray:
        super().set_vals(outputs, targets)
        return 0.5 * np.sum((self._t - self._y)**2)

    def backward(self) -> np.ndarray:
        """Performs a backward pass."""
        return self._y - self._t


class Crossentropy(Loss):
    """Crossentropy loss function"""

    def __call__(self, outputs, targets) -> np.ndarray:
        super().set_vals(outputs, targets)
        return -np.mean(np.log(self._y) * self._t)

    def backward(self) -> np.ndarray:
        """Performs a backward pass."""
        return -1.0 * self._t / (self._y * self._t.size)
