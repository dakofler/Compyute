"""Neural network parameter and buffer classes."""

from typing import Optional

from ..base_tensor import Tensor

__all__ = ["Buffer", "Parameter"]


class Parameter(Tensor):
    """Trainable neural network parameter.

    Parameters
    ----------
    data : Tensor
        Parameter data.
    label : str, optional
        Parameter label, by default None.
    """

    def __init__(self, data: Tensor, label: Optional[str] = None) -> None:
        super().__init__(data.data)
        self.grad = data.grad
        self.label = label


class Buffer(Tensor):
    """Non-trainable neural network buffer variable.

    Parameters
    ----------
    data : Tensor
        Buffer data.
    label : str, optional
        Buffer label, by default None.
    """

    def __init__(self, data: Tensor, label: Optional[str] = None) -> None:
        super().__init__(data.data, requires_grad=False)
        self.label = label
