"""Neural network regularization modules."""

from typing import Optional

from ...base_tensor import Tensor
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
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.
    """

    def __init__(self, p: float = 0.5, label: Optional[str] = None, training: bool = False) -> None:
        super().__init__(label, training)
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self._training:
            return x

        y, self._backward = dropout(x, self.p, self._training)
        return y
