"""Padding functions module"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from walnut.tensor import Tensor


@dataclass
class Padding(ABC):
    """Padding base class."""

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        ...


@dataclass
class Valid(Padding):
    """Applies valid padding using zero-values to a tensor.

    Parameters
    ----------
    x : Tensor
        Tensor, where the padding function is applied to.

    Returns
    -------
    Tensor
        Padded tensor.
    """

    def __call__(self, x: Tensor) -> Tensor:
        return x


@dataclass
class Same(Padding):
    """Applies same padding using zero-values to a tensor.

    Parameters
    ----------
    x : Tensor
        Tensor, where the padding function is applied to.

    Returns
    -------
    Tensor
        Padded tensor.
    """

    width: int

    def __call__(self, x: Tensor, width: int, axis: tuple[int, ...]) -> Tensor:
        pad_axis = tuple(
            (width, width) if ax in axis else (0, 0) for ax in range(x.ndim)
        )
        return Tensor(np.pad(x.data, pad_axis))
