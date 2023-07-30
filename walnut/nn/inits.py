"""Parameter initializations module"""

from dataclasses import dataclass
from abc import ABC, abstractmethod

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ShapeLike


__all__ = [
    "Uniform",
    "Normal",
    "XavierUniform",
    "XavierNormal",
    "KaimingUniform",
    "KaimingNormal",
]

GAINS = {
    "tanh": 5.0 / 3.0,
    "relu": 2**0.5,
}


def compute_gain(act_fn: str) -> float:
    """Computes the gain for a specific activation function.

    Parameters
    ----------
    act_fn : str
        Activation function used.

    Returns
    -------
    float
        Gain value.
    """
    return GAINS.get(act_fn, 1.0)


@dataclass(slots=True)
class InitParams:
    """Parameters for initialization."""

    fan_in: int = 1
    fan_out: int = 1
    act_fn: str = ""


@dataclass(slots=True)
class Init(ABC):
    """Padding base class."""

    params: InitParams

    @abstractmethod
    def __call__(self, shape: ShapeLike) -> Tensor:
        ...


@dataclass(slots=True)
class Uniform(Init):
    """Creates a tensor with values following a uniform distribution."""

    def __call__(self, shape: ShapeLike) -> Tensor:
        """Creates a tensor of a given shape following a uniform distribution.

        Parameters
        ----------
        shape : ShapeLike
            Shape of the new tensor.

        Returns
        -------
        Tensor
            Tensor with random values.
        """
        return tu.randu(shape, -1, 1)


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
class XavierUniform(Init):
    """Creates a tensor with random values as described by Glorot, X. & Bengio, Y. (2010)
    following a uniform distribution."""

    def __call__(self, shape: ShapeLike) -> Tensor:
        """Creates a tensor with random values as described by Glorot, X. & Bengio, Y. (2010)
        following a uniform distribution.

        Parameters
        ----------
        shape : ShapeLike
            Shape of the new tensor.

        Returns
        -------
        Tensor
            Tensor with random values.
        """
        gain = compute_gain(self.params.act_fn)
        fan_in = self.params.fan_in
        fan_out = self.params.fan_out
        bound = gain * (6 / (fan_in + fan_out)) ** 0.5
        return tu.randu(shape, -bound, bound)


@dataclass(slots=True)
class XavierNormal(Init):
    """Creates a tensor with random values as described by Glorot, X. & Bengio, Y. (2010)
    following a normal distribution."""

    def __call__(self, shape: ShapeLike) -> Tensor:
        """Creates a tensor with random values as described by Glorot, X. & Bengio, Y. (2010)
        following a normal distribution.

        Parameters
        ----------
        shape : ShapeLike
            Shape of the new tensor.

        Returns
        -------
        Tensor
            Tensor with random values.
        """
        gain = compute_gain(self.params.act_fn)
        fan_in = self.params.fan_in
        fan_out = self.params.fan_out
        std = gain * (2 / (fan_in + fan_out)) ** 0.5
        return tu.randn(shape, std=std)


@dataclass(slots=True)
class KaimingUniform(Init):
    """Creates a tensor with random values as described by He, K. et al. (2015)
    following a uniform distribution."""

    def __call__(self, shape: ShapeLike) -> Tensor:
        """Creates a tensor with random values as described by He, K. et al. (2015)
        following a uniform distribution.

        Parameters
        ----------
        shape : ShapeLike
            Shape of the new tensor.

        Returns
        -------
        Tensor
            Tensor with random values.
        """
        gain = compute_gain(self.params.act_fn)
        fan_in = self.params.fan_in
        bound = gain * (3 / fan_in) ** 0.5
        return tu.randu(shape, -bound, bound)


@dataclass(slots=True)
class KaimingNormal(Init):
    """Creates a tensor with random values as described by He, K. et al. (2015)
    following a normal distribution."""

    def __call__(self, shape: ShapeLike) -> Tensor:
        """Creates a tensor with random values as described by He, K. et al. (2015)
        following a normal distribution.

        Parameters
        ----------
        shape : ShapeLike
            Shape of the new tensor.

        Returns
        -------
        Tensor
            Tensor with random values.
        """
        gain = compute_gain(self.params.act_fn)
        fan_in = self.params.fan_in
        std = gain / fan_in**0.5
        return tu.randn(shape, std=std)


INITS = {
    "uniform": Uniform,
    "normal": Normal,
    "xavier_uniform": XavierUniform,
    "xavier_normal": XavierNormal,
    "kaiming_uniform": KaimingUniform,
    "kaiming_normal": KaimingNormal,
}
