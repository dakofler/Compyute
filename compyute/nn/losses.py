"""Loss functions module"""

from abc import ABC, abstractmethod
from typing import Callable

from compyute.functional import prod
from compyute.nn.funcional import softmax
from compyute.preprocessing.basic import one_hot_encode
from compyute.tensor import Tensor, ArrayLike


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
        dif = y.float() - t.float()

        def backward() -> ArrayLike:
            """Performs a backward pass."""
            return (dif * 2 / prod(y.shape)).reshape(y.shape).data

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
        t = one_hot_encode(t.int(), y.shape[-1])
        probs = softmax(y.float())

        def backward() -> ArrayLike:
            """Performs a backward pass."""
            return ((probs - t) / prod(y.shape[:-1])).data

        self.backward = backward

        return -((probs + self.eps) * t).sum(-1).log().mean()
