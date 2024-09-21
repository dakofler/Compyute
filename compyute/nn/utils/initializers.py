"""Neural network parameter initializers."""

import math
from collections.abc import Callable
from functools import partial
from typing import Literal, Optional

from ...random.random import normal, uniform
from ...tensor_ops.creation_ops import ones, zeros
from ...tensors import ShapeLike, Tensor
from ..modules.activations import ActivationLike

__all__ = [
    "init_kaiming_normal",
    "init_kaiming_uniform",
    "init_normal",
    "init_ones",
    "init_uniform",
    "init_xavier_normal",
    "init_xavier_uniform",
    "init_zeros",
]


def init_ones(*tensors: Tensor) -> None:
    """Initializes tensors with ones."""
    for t in tensors:
        t.data = ones(t.shape, device=t.device, dtype=t.dtype).data


def init_zeros(*tensors: Tensor) -> None:
    """Initializes tensors with zeros."""
    for t in tensors:
        t.data = zeros(t.shape, device=t.device, dtype=t.dtype).data


def init_normal(*tensors: Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    """Initializes tensors with values following a uniform distribution.

    Parameters
    ----------
    mean : float, optional
        Mean of the normal distribution. Defaults to ``0``.
    std : float, optional
        Standard deviation of the normal distribution. Defaults to ``1``.
    """
    for t in tensors:
        t.data = normal(t.shape, mean, std, device=t.device, dtype=t.dtype).data


def init_uniform(*tensors: Tensor, low: float = -1.0, high: float = 1.0) -> None:
    """Initializes tensors with values following a uniform distribution.

    Parameters
    ----------
    low : float, optional
        Lower bound for random values. Defaults to ``-1``.
    high : float, optional
        Upper bound for random values. Defaults to ``1``.
    """
    for t in tensors:
        t.data = uniform(t.shape, low, high, device=t.device, dtype=t.dtype).data


def init_kaiming_normal(
    *tensors: Tensor, activation: Optional[ActivationLike] = None
) -> None:
    """Normal initializer as described by
    `He et al., 2015 <https://arxiv.org/pdf/1502.01852>`_.

    Parameters
    ----------
    activation : _ActivationLike
        Activation function to infer the gain value for.
    """
    gain = _get_gain(activation)
    for t in tensors:
        fan_in = get_fan_in(t.shape)
        std = gain / math.sqrt(fan_in)
        t.data = normal(t.shape, 0, std, device=t.device, dtype=t.dtype).data


def init_kaiming_uniform(
    *tensors: Tensor, activation: Optional[ActivationLike] = None
) -> None:
    """Uniform initializer as described by
    `He et al., 2015 <https://arxiv.org/pdf/1502.01852>`_.

    Parameters
    ----------
    activation : _ActivationLike
        Activation function to infer the gain value for.
    """
    gain = _get_gain(activation)
    for t in tensors:
        fan_in = get_fan_in(t.shape)
        k = gain * math.sqrt(3 / fan_in)
        t.data = uniform(t.shape, -k, k, device=t.device, dtype=t.dtype).data


def init_xavier_normal(
    *tensors: Tensor, activation: Optional[ActivationLike] = None
) -> None:
    """Normal initializer as described by
    `Glorot et al., 2010 <https://proceedings.mlr.press/v9/glorot10a.html>`_.

    Parameters
    ----------
    activation : _ActivationLike
        Activation function to infer the gain value for.
    """
    gain = _get_gain(activation)
    for t in tensors:
        fan_in = get_fan_in(t.shape)
        fan_out = get_fan_out(t.shape)
        std = gain * math.sqrt(2 / (fan_in + fan_out))
        t.data = normal(t.shape, 0, std, device=t.device, dtype=t.dtype).data


def init_xavier_uniform(
    *tensors: Tensor, activation: Optional[ActivationLike] = None
) -> None:
    """Uniform initializer as described by
    `Glorot et al., 2010 <https://proceedings.mlr.press/v9/glorot10a.html>`_.

    Parameters
    ----------
    activation : _ActivationLike
        Activation function to infer the gain value for.
    """
    gain = _get_gain(activation)
    for t in tensors:
        fan_in = get_fan_in(t.shape)
        fan_out = get_fan_out(t.shape)
        k = gain * math.sqrt(6 / (fan_in + fan_out))
        t.data = uniform(t.shape, -k, k, device=t.device, dtype=t.dtype).data


InitializerLike = Literal[
    "default",
    "kaiming_normal",
    "kaiming_uniform",
    "normal",
    "uniform",
    "xavier_normal",
    "xavier_uniform",
    "zeros",
    "ones",
]

INITIALIZERS: dict[str, Callable[..., None]] = {
    "kaiming_normal": init_kaiming_normal,
    "kaiming_uniform": init_kaiming_uniform,
    "normal": init_normal,
    "uniform": init_uniform,
    "xavier_normal": init_xavier_normal,
    "xavier_uniform": init_xavier_uniform,
    "zeros": init_zeros,
    "ones": init_ones,
}

GAINS = {
    "sigmoid": 1,
    "tanh": 5 / 3,
    "relu": math.sqrt(2),
    "leaky_relu": math.sqrt(2 / (1 + 0.1**2)),
    "selu": 3 / 4,
}


def _get_gain(activation: Optional[ActivationLike] = None) -> float:
    """Returns the gain value to use for initializing values."""
    if activation is None:
        return 1
    return GAINS.get(activation, 1)


def get_initializer(
    initializer: InitializerLike, activation: ActivationLike
) -> Callable[..., None]:
    """Returns an instance of an initializer."""
    if initializer not in INITIALIZERS:
        raise ValueError(f"Unknown initializer: {initializer}.")
    elif initializer in {
        "xavier_normal",
        "xavier_uniform",
        "kaiming_normal",
        "kaiming_uniform",
    }:
        return partial(INITIALIZERS[initializer], activation=activation)
    return INITIALIZERS[initializer]


def get_fan_in(shape: ShapeLike) -> int:
    """Returns the fan-in value for a given shape."""
    return math.prod(shape[1:])


def get_fan_out(shape: ShapeLike) -> int:
    """Returns the fan-out value for a given shape."""
    return math.prod((shape[0],) + shape[2:])
