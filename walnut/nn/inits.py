"""Parameter initializations module"""

from dataclasses import dataclass
from abc import ABC, abstractmethod

from walnut import tensor
from walnut.tensor import Tensor


KAIMING_GAINS = {
    "tanh": 5.0 / 3.0,
    "relu": 2**0.5,
}


@dataclass
class InitParams:
    """Parameters for initialization."""

    fan_mode: int = 1
    act_fn: str | None = ""


@dataclass
class Init(ABC):
    """Padding base class."""

    params: InitParams

    @abstractmethod
    def __call__(self, shape: tuple[int, ...]) -> Tensor:
        ...


@dataclass
class Random(Init):
    """Creates a tensor of a given shape following a normal distribution."""

    def __call__(self, shape: tuple[int, ...]) -> Tensor:
        """Creates a tensor of a given shape following a normal distribution.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the new tensor.

        Returns
        -------
        Tensor
            Tensor with random values.
        """
        return tensor.randn(shape)


@dataclass
class KaimingHe(Init):
    """Creates a tensor of a given shape with values using Kaiming He initialization."""

    def __call__(self, shape: tuple[int, ...]) -> Tensor:
        """Creates a tensor of a given shape with values using Kaiming He initialization.

        Parameters
        ----------
        shape : tuple[int, ...]
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
        return tensor.randn(shape) * gain / self.params.fan_mode**0.5


INITS = {"random": Random, "kaiming_he": KaimingHe}
