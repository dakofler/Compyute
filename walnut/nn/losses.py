"""Loss functions module"""

from abc import ABC, abstractmethod
from typing import Callable

from walnut.tensor import Tensor, ArrayLike
import walnut.tensor_utils as tu
from walnut.nn.funcional import softmax
from walnut.preprocessing.basic import one_hot_encode


__all__ = ["MSE", "Crossentropy"]


class Loss(ABC):
    """Loss base class."""

    def __init__(self):
        self.backward: Callable[[], ArrayLike] | None = None

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
        dif = y - t.astype(y.dtype)

        def backward() -> ArrayLike:
            """Performs a backward pass."""
            return (dif * 2.0 / tu.prod(y.shape)).data

        self.backward = backward

        return (dif**2).mean()


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
        t.int()
        t = one_hot_encode(t, y.shape[-1]).astype(y.dtype)
        probs = softmax(y)

        def backward() -> ArrayLike:
            """Performs a backward pass."""
            return ((probs - t) / tu.prod(y.shape[:-1])).data

        self.backward = backward

        return (-((probs + self.eps).log() * t).sum(axis=-1)).mean()
