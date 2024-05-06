"""Regularization layers module"""

from typing import Optional
from ..module import Module
from ....random import multinulli
from ....tensor import Tensor


__all__ = ["Dropout"]


class Dropout(Module):
    """Dropout layer used to randomly reduce information and avoid overfitting."""

    def __init__(self, p: float = 0.5, label: Optional[str] = None) -> None:
        """Dropout layer used to randomly reduce information and avoid overfitting.

        Parameters
        ----------
        p : float, optional
            Probability of values being set to zero, by default 0.5.
        label: str, optional
            Module label.
        """
        super().__init__(label)
        self.p = p

    def __repr__(self) -> str:
        label = self.label
        p = self.p
        return f"{label}({p=})"

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            d_map = multinulli(self.p, x.shape, device=self.device)
            y = x * d_map / (1 - self.p)

            # use d_map as mask for grads
            self.backward_fn = lambda dy: dy * d_map / (1 - self.p)

        else:
            y = x

        return y
