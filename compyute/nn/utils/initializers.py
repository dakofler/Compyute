"""Neural network parameter initializers."""

import math
from abc import ABC, abstractmethod
from typing import Literal, Optional

from ...base_tensor import ShapeLike, Tensor
from ...random.random import normal, uniform
from ...tensor_ops.creating import ones, zeros
from ..modules.activations import _ActivationLike
from ..parameter import Parameter

__all__ = [
    "KaimingNormal",
    "KaimingUniform",
    "Normal",
    "Uniform",
    "XavierNormal",
    "XavierUniform",
]


class Initializer(ABC):
    """Initializer base class."""

    @abstractmethod
    def __call__(self, x: Tensor | Parameter) -> None:
        """Fills a tensor with values.

        Parameters
        ----------
        x : Tensor | Parameter
            Tensor or parameter to fill with values.
        """


class Ones(Initializer):
    """Initializes a tensor with ones."""

    def __call__(self, x: Tensor | Parameter) -> None:
        x.data = ones(x.shape, x.dtype, x.device).data


class Zeros(Initializer):
    """Initializes a tensor with zeros."""

    def __call__(self, x: Tensor | Parameter) -> None:
        x.data = zeros(x.shape, x.dtype, x.device).data


class Normal(Initializer):
    """Initializes a tensor with values following a uniform distribution.

    Parameters
    ----------
    mean : float, optional
        Mean of the normal distribution. Defaults to ``0``.
    std : float, optional
        Standard deviation of the normal distribution. Defaults to ``1``.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, x: Tensor | Parameter) -> None:
        x.data = normal(x.shape, self.mean, self.std, x.dtype, x.device).data


class Uniform(Initializer):
    """Initializes a tensor with values following a uniform distribution.

    Parameters
    ----------
    low : float, optional
        Lower bound for random values. Defaults to ``-1``.
    high : float, optional
        Upper bound for random values. Defaults to ``1``.
    """

    def __init__(self, low: float = -1.0, high: float = 1.0) -> None:
        self.low = low
        self.high = high

    def __call__(self, x: Tensor | Parameter) -> None:
        x.data = uniform(x.shape, self.low, self.high, x.dtype, x.device).data


class KaimingNormal(Initializer):
    """Normal initializer as described by
    `He et al., 2015 <https://arxiv.org/pdf/1502.01852>`_.

    Parameters
    ----------
    activation : _ActivationLike
        Activation function to infer the gain value for.
    """

    def __init__(self, activation: Optional[_ActivationLike] = None) -> None:
        self.gain = get_gain(activation)

    def __call__(self, x: Tensor | Parameter) -> None:
        fan_in = get_fan_in(x.shape)
        std = self.gain / math.sqrt(fan_in)
        x.data = normal(x.shape, 0, std, x.dtype, x.device).data


class KaimingUniform(Initializer):
    """Uniform initializer as described by
    `He et al., 2015 <https://arxiv.org/pdf/1502.01852>`_.

    Parameters
    ----------
    activation : _ActivationLike
        Activation function to infer the gain value for.
    """

    def __init__(self, activation: Optional[_ActivationLike] = None) -> None:
        self.gain = get_gain(activation)

    def __call__(self, x: Tensor | Parameter) -> None:
        fan_in = get_fan_in(x.shape)
        k = self.gain * math.sqrt(3 / fan_in)
        x.data = uniform(x.shape, -k, k, x.dtype, x.device).data


class XavierNormal(Initializer):
    """Normal initializer as described by
    `Glorot et al., 2010 <https://proceedings.mlr.press/v9/glorot10a.html>`_.

    Parameters
    ----------
    activation : _ActivationLike
        Activation function to infer the gain value for.
    """

    def __init__(self, activation: Optional[_ActivationLike] = None) -> None:
        self.gain = get_gain(activation)

    def __call__(self, x: Tensor | Parameter) -> None:
        fan_in = get_fan_in(x.shape)
        fan_out = _get_fan_out(x.shape)
        std = self.gain * math.sqrt(2 / (fan_in + fan_out))
        x.data = normal(x.shape, 0, std, x.dtype, x.device).data


class XavierUniform(Initializer):
    """Uniform initializer as described by
    `Glorot et al., 2010 <https://proceedings.mlr.press/v9/glorot10a.html>`_.

    Parameters
    ----------
    activation : _ActivationLike
        Activation function to infer the gain value for.
    """

    def __init__(self, activation: Optional[_ActivationLike] = None) -> None:
        self.gain = get_gain(activation)

    def __call__(self, x: Tensor | Parameter) -> None:
        fan_in = get_fan_in(x.shape)
        fan_out = _get_fan_out(x.shape)
        k = self.gain * math.sqrt(6 / (fan_in + fan_out))
        x.data = uniform(x.shape, -k, k, x.dtype, x.device).data


_InitializerLike = (
    Initializer
    | Literal[
        "kaiming_normal",
        "kaiming_uniform",
        "normal",
        "uniform",
        "xavier_normal",
        "xavier_uniform",
        "zeros",
        "ones",
    ]
)

INITIALIZERS = {
    "kaiming_normal": KaimingNormal,
    "kaiming_uniform": KaimingUniform,
    "normal": Normal,
    "uniform": Uniform,
    "xavier_normal": XavierNormal,
    "xavier_uniform": XavierUniform,
    "zeros": Zeros,
    "ones": Ones,
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
) -> Initializer:
    """Returns an instance of an initializer."""
    if isinstance(initializer, Initializer):
        return initializer
    if initializer not in INITIALIZERS:
        raise ValueError(f"Unknown initializer: {initializer}.")
    if initializer in {
        "xavier_normal",
        "xavier_uniform",
        "kaiming_normal",
        "kaiming_uniform",
    }:
        return INITIALIZERS[initializer](activation)
    return INITIALIZERS[initializer]()


def get_fan_in(shape: ShapeLike) -> int:
    """Returns the fan-in value for a given shape."""
    return math.prod((shape[0],) + shape[2:])


def _get_fan_out(shape: ShapeLike) -> int:
    """Returns the fan-out value for a given shape."""
    return math.prod((shape[1],) + shape[2:])
