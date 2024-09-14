"""Neural network regularization modules."""

from typing import Optional

from ...tensors import Tensor
from ..functional.regularizations import FDropout
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

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return FDropout.forward(self.fcache, x, self.p, self._is_training)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return FDropout.backward(self.fcache, dy)
