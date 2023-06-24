"""loss functions module"""

import numpy as np
from numpynn.tensor import Tensor

class Loss():
    """Loss base class."""

    def __init__(self) -> None:
        self.loss = 0
        self._y = None
        self._t = None

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
