"""Parameter initializations module"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import math

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ShapeLike


__all__ = ["Normal", "KaimingHe"]


KAIMING_GAINS = {
    "tanh": 5.0 / 3.0,
    "relu": 2**0.5,
}


@dataclass(slots=True)
class InitParams:
    """Parameters for initialization."""

    fan_in: int = 1
    fan_out: int = 1
    act_fn: str | None = ""


@dataclass(slots=True)
class Init(ABC):
    """Padding base class."""

    params: InitParams

    @abstractmethod
    def __call__(self, shape: ShapeLike) -> Tensor:
        ...


@dataclass(slots=True)
class Normal(Init):
    """Creates a tensor with values following a normal distribution."""

    def __call__(self, shape: ShapeLike) -> Tensor:
        """Creates a tensor of a given shape following a normal distribution.

        Parameters
        ----------
        shape : ShapeLike
            Shape of the new tensor.

        Returns
        -------
        Tensor
            Tensor with random values.
        """
        return tu.randn(shape)


@dataclass(slots=True)
class KaimingHe(Init):
    """Creates a tensor of a given shape with values using Kaiming He initialization."""

    def __call__(self, shape: ShapeLike) -> Tensor:
        """Creates a tensor with random values as described by He, K. et al. (2015).

        Parameters
        ----------
        shape : ShapeLike
            Shape of the new tensor.

        Returns
        -------
        Tensor
            Tensor with random values.
        """
        gain = (
            KAIMING_GAINS.get(self.params.act_fn, 1.0)
            if self.params.act_fn is not None
            else 1.0
        )
        return tu.randn(shape) * gain / self.params.fan_in**0.5


@dataclass(slots=True)
class KUniform(Init):
    """Creates a tensor of a given shape with values following a scaled uniform distribution."""

    def __call__(self, shape: ShapeLike) -> Tensor:
        """Creates a tensor with random values following a scaled uniform distribution.

        Parameters
        ----------
        shape : ShapeLike
            Shape of the new tensor.

        Returns
        -------
        Tensor
            Tensor with random values.
        """
        scale = math.sqrt(1.0 / self.params.fan_in)
        return tu.uniform(-scale, scale, shape)


@dataclass(slots=True)
class Glorot(Init):
    """Creates a tensor of a given shape with values following a scaled uniform distribution."""

    def __call__(self, shape: ShapeLike) -> Tensor:
        """Creates a tensor with random values following a scaled uniform distribution.

        Parameters
        ----------
        shape : ShapeLike
            Shape of the new tensor.

        Returns
        -------
        Tensor
            Tensor with random values.
        """
        scale = math.sqrt(6.0 / (self.params.fan_in + self.params.fan_out))
        return tu.uniform(-scale, scale, shape)


INITS = {
    "normal": Normal,
    "kaiming_he": KaimingHe,
    "kuniform": KUniform,
    "glorot": Glorot,
}
