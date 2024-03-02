"""Regularization layers module"""

from __future__ import annotations

from compyute.tensor_functions import random_choice
from compyute.tensor import Tensor, ArrayLike
from compyute.nn.module import Module


__all__ = ["Dropout"]


class Dropout(Module):
    """Dropout layer used to randomly reduce information and avoid overfitting."""

    def __init__(self, p: float = 0.5) -> None:
        """Dropout layer used to randomly reduce information and avoid overfitting.

        Parameters
        ----------
        p : float, optional
            Probability of values being set to zero, by default 0.5.
        """
        super().__init__()
        self.p = p

    def __repr__(self) -> str:
        name = self.__class__.__name__
        p = self.p
        return f"{name}({p=})"

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            choices = Tensor([0, 1], dtype=x.dtype)
            probs = Tensor([self.p, 1.0 - self.p], dtype=x.dtype)
            d_map = random_choice(choices, probs, x.shape, self.device)
            y = x * d_map / (1.0 - self.p)

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)

                # use d_map as mask for grads
                return dy * d_map.data / (1.0 - self.p)

            self.backward = backward

        else:
            y = x

        self.set_y(y)
        return y
