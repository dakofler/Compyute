"""Parameter initializations module"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from walnut import tensor
from walnut.tensor import Tensor


@dataclass
class Init(ABC):
    """Padding base class."""

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

    fan_mode: int
    act_fn: str
    GAINS: dict[str, float] = field(
        default_factory=lambda: (
            {
                "NoneType": 1.0,
                "Sigmoid": 1.0,
                "Tanh": 5.0 / 3.0,
                "Relu": 2**0.5,
                "Softmax": 1.0,
            }
        )
    )

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
        gain = self.GAINS.get(self.act_fn, 1)
        return tensor.randn(shape) * gain / self.fan_mode**0.5
