"""Neural network parameter initializers."""

import math
from functools import partial
from typing import Callable, Literal, Optional

from ...random.random import normal as _normal
from ...random.random import uniform as _uniform
from ...tensor_ops.creating import ones as _ones
from ...tensor_ops.creating import zeros as _zeros
from ...tensors import ShapeLike, Tensor
from ..modules.activations import _ActivationLike

__all__ = [
    "kaiming_normal",
    "kaiming_uniform",
    "normal",
    "uniform",
    "xavier_normal",
    "xavier_uniform",
]


def ones(*tensors: Tensor) -> None:
    """Initializes tensors with ones."""
    for t in tensors:
        t.data = _ones(t.shape, t.device, t.dtype).data


def zeros(*tensors: Tensor) -> None:
    """Initializes tensors with zeros."""
    for t in tensors:
        t.data = _zeros(t.shape, t.device, t.dtype).data


def normal(*tensors: Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    """Initializes tensors with values following a uniform distribution.

    Parameters
    ----------
    mean : float, optional
        Mean of the normal distribution. Defaults to ``0``.
    std : float, optional
        Standard deviation of the normal distribution. Defaults to ``1``.
    """
    for t in tensors:
        t.data = _normal(t.shape, mean, std, t.device, t.dtype).data


def uniform(*tensors: Tensor, low: float = -1.0, high: float = 1.0) -> None:
    """Initializes tensors with values following a uniform distribution.

    Parameters
    ----------
    low : float, optional
        Lower bound for random values. Defaults to ``-1``.
    high : float, optional
        Upper bound for random values. Defaults to ``1``.
    """
    for t in tensors:
        t.data = _uniform(t.shape, low, high, t.device, t.dtype).data


def kaiming_normal(
    *tensors: Tensor, activation: Optional[_ActivationLike] = None
) -> None:
    """Normal initializer as described by
    `He et al., 2015 <https://arxiv.org/pdf/1502.01852>`_.

    Parameters
    ----------
    activation : _ActivationLike
        Activation function to infer the gain value for.
    """
    gain = get_gain(activation)
    for t in tensors:
        fan_in = _get_fan_in(t.shape)
        std = gain / math.sqrt(fan_in)
        t.data = _normal(t.shape, 0, std, t.device, t.dtype).data


def kaiming_uniform(
    *tensors: Tensor, activation: Optional[_ActivationLike] = None
) -> None:
    """Uniform initializer as described by
    `He et al., 2015 <https://arxiv.org/pdf/1502.01852>`_.

    Parameters
    ----------
    activation : _ActivationLike
        Activation function to infer the gain value for.
    """
    gain = get_gain(activation)
    for t in tensors:
        fan_in = _get_fan_in(t.shape)
        k = gain * math.sqrt(3 / fan_in)
        t.data = _uniform(t.shape, -k, k, t.device, t.dtype).data


def xavier_normal(
    *tensors: Tensor, activation: Optional[_ActivationLike] = None
) -> None:
    """Normal initializer as described by
    `Glorot et al., 2010 <https://proceedings.mlr.press/v9/glorot10a.html>`_.

    Parameters
    ----------
    activation : _ActivationLike
        Activation function to infer the gain value for.
    """
    gain = get_gain(activation)
    for t in tensors:
        fan_in = _get_fan_in(t.shape)
        fan_out = _get_fan_out(t.shape)
        std = gain * math.sqrt(2 / (fan_in + fan_out))
        t.data = _normal(t.shape, 0, std, t.device, t.dtype).data


def xavier_uniform(
    *tensors: Tensor, activation: Optional[_ActivationLike] = None
) -> None:
    """Uniform initializer as described by
    `Glorot et al., 2010 <https://proceedings.mlr.press/v9/glorot10a.html>`_.

    Parameters
    ----------
    activation : _ActivationLike
        Activation function to infer the gain value for.
    """
    gain = get_gain(activation)
    for t in tensors:
        fan_in = _get_fan_in(t.shape)
        fan_out = _get_fan_out(t.shape)
        k = gain * math.sqrt(6 / (fan_in + fan_out))
        t.data = _uniform(t.shape, -k, k, t.device, t.dtype).data


_InitializerLike = Literal[
    "kaiming_normal",
    "kaiming_uniform",
    "normal",
    "uniform",
    "xavier_normal",
    "xavier_uniform",
    "zeros",
    "ones",
]

INITIALIZERS = {
    "kaiming_normal": kaiming_normal,
    "kaiming_uniform": kaiming_uniform,
    "normal": normal,
    "uniform": uniform,
    "xavier_normal": xavier_normal,
    "xavier_uniform": xavier_uniform,
    "zeros": zeros,
    "ones": ones,
}

GAINS = {
    "sigmoid": 1,
    "tanh": 5 / 3,
    "relu": math.sqrt(2),
    "leaky_relu": math.sqrt(2 / (1 + 0.1**2)),
    "selu": 3 / 4,
}


def get_gain(activation: Optional[_ActivationLike] = None) -> float:
    """Returns the gain value to use for initializing values.

    Parameters
    ----------
    activation : _ActivationLike, optional
        Activation function to infer the gain value for. Defaults to ``None``.
        If ``None``, a value of ``1`` is returned.

    Returns
    -------
    float
        Gain value.
    """
    if activation is None:
        return 1
    return GAINS.get(activation, 1)


def get_initializer(
    initializer: _InitializerLike, activation: _ActivationLike
) -> Callable:
    """Returns an instance of an initializer."""
    if initializer not in INITIALIZERS:
        raise ValueError(f"Unknown initializer: {initializer}.")
    if initializer in {
        "xavier_normal",
        "xavier_uniform",
        "kaiming_normal",
        "kaiming_uniform",
    }:
        return partial(INITIALIZERS[initializer], activation=activation)
    return INITIALIZERS[initializer]


def _get_fan_in(shape: ShapeLike) -> int:
    """Returns the fan-in value for a given shape."""
    return math.prod(shape[1:])


def _get_fan_out(shape: ShapeLike) -> int:
    """Returns the fan-out value for a given shape."""
    return math.prod((shape[0],) + shape[2:])
