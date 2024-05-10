"""Parameter initializations module"""

from .. import random
from ..tensor_f import prod
from ..basetensor import Tensor
from ..types import ShapeLike


__all__ = [
    "get_gain",
    "uniform",
    "normal",
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
]

GAINS = {
    "linear": 1,
    "conv1d": 1,
    "conv2d": 1,
    "sigmoid": 1,
    "tanh": 5 / 3,
    "relu": 2**0.5,
    "leaky_relu": (2 / (1 + 0.1**2)) ** 0.5,
    "selu": 3 / 4,
}


def get_gain(act_fn: str) -> float:
    """Returns the gain for a specific activation function.

    Parameters
    ----------
    act_fn : str
        Activation function used.

    Returns
    -------
    float
        Gain value.
    """
    return GAINS.get(act_fn, 1)


def get_fan_in(shape: ShapeLike) -> int:
    """Computes the fan_in for a given shape.

    Parameters
    ----------
    shape : ShapeLike
        Shape of a tensor.

    Returns
    -------
    int
        Fan_in value.
    """
    return prod((shape[0],) + shape[2:])


def get_fan_out(shape: ShapeLike) -> int:
    """Computes the fan_out for a given shape.

    Parameters
    ----------
    shape : ShapeLike
        Shape of a tensor.

    Returns
    -------
    int
        Fan_out value.
    """
    return prod((shape[1],) + shape[2:])


def uniform(shape: ShapeLike, low: float = 0.0, high: float = 1.0) -> Tensor:
    """Returns a tensor of a given shape with values following a uniform distribution.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the new tensor.
    low : float, optional
        Lower bound for random values, by default 0.
    high : float, optional
        Upper bound for random values, by default 1.

    Returns
    -------
    Tensor
        Tensor with random values.
    """
    return random.uniform(shape, low, high)


def normal(shape: ShapeLike, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Returns a tensor of a given shape with values following a normal distribution.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the new tensor.
    mean : float, optional
        Mean of random values, by default 0.
    std : float, optional
        Standard deviation of random values, by default 1.

    Returns
    -------
    Tensor
        Tensor with random values.
    """
    return random.normal(shape, mean, std)


def xavier_uniform(shape: ShapeLike, gain: float = 1.0) -> Tensor:
    """Returns a tensor with random values as described by Glorot, X. & Bengio, Y. (2010)
    following a uniform distribution.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the new tensor.
    gain : float, optional
        Gain used in the initialization, by default 1.0.

    Returns
    -------
    Tensor
        Tensor with random values.
    """
    fan_in = get_fan_in(shape)
    fan_out = get_fan_out(shape)
    bound = gain * (6 / (fan_in + fan_out)) ** 0.5
    return random.uniform(shape, -bound, bound)


def xavier_normal(shape: ShapeLike, gain: float = 1.0) -> Tensor:
    """Returns a tensor with random values as described by Glorot, X. & Bengio, Y. (2010)
    following a normal distribution.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the new tensor.
    gain : float, optional
        Gain used in the initialization, by default 1.0.

    Returns
    -------
    Tensor
        Tensor with random values.
    """
    fan_in = get_fan_in(shape)
    fan_out = get_fan_out(shape)
    std = gain * (2 / (fan_in + fan_out)) ** 0.5
    return random.normal(shape, std=std)


def kaiming_uniform(shape: ShapeLike, gain: float = 1.0) -> Tensor:
    """Returns a tensor with random values as described by He, K. et al. (2015)
    following a uniform distribution.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the new tensor.
    gain : float, optional
        Gain used in the initialization, by default 1.0.

    Returns
    -------
    Tensor
        Tensor with random values.
    """
    fan_in = get_fan_in(shape)
    bound = gain * (3 / fan_in) ** 0.5
    return random.uniform(shape, -bound, bound)


def kaiming_normal(shape: ShapeLike, gain: float = 1.0) -> Tensor:
    """Returns a tensor with random values as described by He, K. et al. (2015)
    following a normal distribution.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the new tensor.
    gain : float, optional
        Gain used in the initialization, by default 1.0.

    Returns
    -------
    Tensor
        Tensor with random values.
    """
    fan_in = get_fan_in(shape)
    std = gain / fan_in**0.5
    return random.normal(shape, std=std)
