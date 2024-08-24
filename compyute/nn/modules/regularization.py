"""Neural network regularization modules."""

from typing import Optional

from ...tensors import Tensor
from ..functional.regularizations import dropout
from .module import Module

__all__ = ["Dropout"]


class Dropout(Module):
    """Dropout layer used for regularization.

    Parameters
    ----------
    p : float, optional
        Probability of values being set to zero. Defaults to ``0.5``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(self, p: float = 0.5, label: Optional[str] = None) -> None:
        super().__init__(label)
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self._is_training:
            return x

        y, self._backward = dropout(x, self.p, self._is_training)
        return y
