"""Neural network parameter module"""

from typing import Optional
from ..basetensor import Tensor


__all__ = ["Parameter"]


class Parameter(Tensor):
    """Trainable neural network parameter."""

    def __init__(
        self,
        data: Tensor,
        label: Optional[str] = None,
    ) -> None:
        """Trainable neural network parameter.

        Parameters
        ----------
        data : Tensor
            Parameter data.
        label: str, optional
            Parameter label, by default None.
        """
        super().__init__(data.data)
        self.label = label
