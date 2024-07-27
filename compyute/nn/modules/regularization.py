"""Neural network regularization modules."""

from typing import Optional

from ...base_tensor import Tensor
from ...random.random import multinulli
from .module import Module

__all__ = ["Dropout"]


class Dropout(Module):
    """Dropout layer used for regularization.

    Parameters
    ----------
    p : float, optional
        Probability of values being set to zero, by default 0.5.
    label : str, optional
        Module label.
    training : bool, optional
        Whether the module should be in training mode, by default False.
    """

    def __init__(self, p: float = 0.5, label: Optional[str] = None, training: bool = False) -> None:
        super().__init__(label, training)
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self._training:
            return x

        dropout_map = multinulli(self.p, x.shape, device=self.device) / (1 - self.p)
        y = x * dropout_map
        self._backward = lambda dy: dy * dropout_map
        return y
