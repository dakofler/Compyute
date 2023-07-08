"""Padding functions module"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from walnut.tensor import Tensor


@dataclass(slots=True)
class PaddingParams:
    """Parameters for padding."""

    width: int = 1
    axis: tuple[int, ...] = (-1, -2)


@dataclass(slots=True)
class Padding(ABC):
    """Padding base class."""

    params: PaddingParams

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        ...


@dataclass(slots=True)
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


@dataclass(slots=True)
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

    def __call__(self, x: Tensor) -> Tensor:
        wdt = self.params.width
        axis = self.params.axis
        pad_axis = tuple((wdt, wdt) if ax in axis else (0, 0) for ax in range(x.ndim))
        return Tensor(np.pad(x.data, pad_axis))


PADDINGS = {"valid": Valid, "same": Same}
