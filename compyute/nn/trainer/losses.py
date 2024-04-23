"""Loss functions module"""

from abc import ABC, abstractmethod
from typing import Callable, Optional
from ..functional import softmax
from ...tensor_f import prod
from ...preprocessing.basic import one_hot_encode
from ...tensor import Tensor


__all__ = ["BinaryCrossentropy", "Crossentropy", "MSE"]


class Loss(ABC):
    """Loss base class."""

    def __init__(self):
        self.backward: Optional[Callable[[], Tensor]] = None

    @abstractmethod
    def __call__(self, y: Tensor, t: Tensor) -> Tensor: ...


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
        y = y.float()
        t = t.float()

        self.backward = lambda: (y - t) * 2 / prod(y.shape)

        return ((y - t) ** 2).mean()


class Crossentropy(Loss):
    """Computes the crossentropy loss."""

    def __init__(self, eps: float = 1e-8):
        """Computes the crossentropy loss.

        Parameters
        ----------
        eps : float, optional
            Constant used for numerical stability, by default 1e-8.
        """
        super().__init__()
        self.eps = eps

    def __call__(self, y: Tensor, t: Tensor) -> Tensor:
        """Computes the crossentropy loss.

        Parameters
        ----------
        y : Tensor
            A model's logits.
        t : Tensor
            Target class labels.

        Returns
        -------
        Tensor
            Crossentropy loss.
        """
        y = y.float()
        t = t.int()

        t = one_hot_encode(t, y.shape[-1])
        probs = softmax(y)

        self.backward = lambda: (probs - t) / prod(y.shape[:-1])

        return -((probs + self.eps) * t).sum(-1).log().mean()


class BinaryCrossentropy(Loss):
    """Computes the binary crossentropy loss."""

    def __call__(self, y: Tensor, t: Tensor) -> Tensor:
        """Computes the binary crossentropy loss.

        Parameters
        ----------
        y : Tensor
            A model's logits.
        t : Tensor
            Target class labels.

        Returns
        -------
        Tensor
            Crossentropy loss.
        """
        y = y.float()
        t = t.float()
        c = 100

        self.backward = lambda: (-t / y + (1 - t) / (1 - y)) / prod(y.shape)

        return -(t * y.log().clip(-c, c) + (1 - t) * (1 - y).log().clip(-c, c)).mean()


LOSSES = {
    "binary_crossentropy": BinaryCrossentropy,
    "crossentropy": Crossentropy,
    "mse": MSE,
}


def get_loss(loss: Loss | str) -> Loss:
    """Returns an instance of a loss function."""
    if isinstance(loss, Loss):
        return loss
    if loss not in LOSSES.keys():
        raise ValueError(f"Unknown loss function {loss}.")
    return LOSSES[loss]()
