"""Neural network parameter and buffer classes."""

from __future__ import annotations

from typing import Optional

from ..base_tensor import Tensor
from ..typing import is_float

__all__ = ["Buffer", "Parameter"]


class Parameter(Tensor):
    """Trainable neural network parameter.

    Parameters
    ----------
    data : Tensor
        Data to initialize the parameter. Must be of type ``float``.

    Raises
    ------
    TypeError
        If an invalid data type is provided.
    """

    def __init__(self, data: Tensor) -> None:
        if not is_float(data.dtype):
            raise TypeError("Invalid data type for parameter. Must be float.")
        super().__init__(data.data)


def update_parameter_grad(
    parameter: Optional[Parameter], grad: Optional[Tensor]
) -> None:
    """Updates the parameter gradients."""
    if parameter and grad:
        parameter.grad += grad


class Buffer(Tensor):
    """Neural network buffer variable.

    Parameters
    ----------
    data : Tensor
        Data to initialize the buffer.
    """

    def __init__(self, data: Tensor) -> None:
        super().__init__(data.data)
