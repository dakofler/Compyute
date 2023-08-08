"""Parameter initializations module"""

import numpy as np

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ShapeLike


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
    "tanh": 5.0 / 3.0,
    "relu": 2**0.5,
    "leaky_relu": (2.0 / (1 + 0.1**2)) ** 0.5,
    "selu": 3.0 / 4.0,
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
    return GAINS.get(act_fn, 1.0)


def compute_fan_in(shape: ShapeLike) -> int:
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
    return int(shape[0] * np.prod(shape[2:]))


def compute_fan_out(shape: ShapeLike) -> int:
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
    return int(shape[1] * np.prod(shape[2:]))


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
    return tu.randu(shape, low, high)


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
    return tu.randn(shape, mean, std)


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
    fan_in = compute_fan_in(shape)
    fan_out = compute_fan_out(shape)
    bound = gain * (6.0 / (fan_in + fan_out)) ** 0.5
    return tu.randu(shape, -bound, bound)


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
    fan_in = compute_fan_in(shape)
    fan_out = compute_fan_out(shape)
    std = gain * (2.0 / (fan_in + fan_out)) ** 0.5
    return tu.randn(shape, std=std)


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
    fan_in = compute_fan_in(shape)
    bound = gain * (3.0 / fan_in) ** 0.5
    return tu.randu(shape, -bound, bound)


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
    fan_in = compute_fan_in(shape)
    std = gain / fan_in**0.5
    return tu.randn(shape, std=std)
