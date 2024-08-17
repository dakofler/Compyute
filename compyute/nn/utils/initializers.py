"""Neural network parameter initializers."""

import math
import operator
from abc import ABC, abstractmethod
from functools import reduce
from typing import Literal, Optional

from ...base_tensor import Tensor, _ShapeLike
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
    def __call__(self, tensor: Tensor | Parameter) -> None:
        """Fills a tensor with values.

        Parameters
        ----------
        tensor : Tensor | Parameter
            Tensor or parameter to fill with values.
        """


class Ones(Initializer):
    """Initializes a tensor with ones."""

    def __call__(self, tensor: Tensor | Parameter) -> None:
        tensor.data = ones(tensor.shape, tensor.dtype, tensor.device).data


class Zeros(Initializer):
    """Initializes a tensor with zeros."""

    def __call__(self, tensor: Tensor | Parameter) -> None:
        tensor.data = zeros(tensor.shape, tensor.dtype, tensor.device).data


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

    def __call__(self, tensor: Tensor | Parameter) -> None:
        tensor.data = normal(
            tensor.shape, self.mean, self.std, tensor.dtype, tensor.device
        ).data


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

    def __call__(self, tensor: Tensor | Parameter) -> None:
        tensor.data = uniform(
            tensor.shape, self.low, self.high, tensor.dtype, tensor.device
        ).data


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

    def __call__(self, tensor: Tensor | Parameter) -> None:
        fan_in = get_fan_in(tensor.shape)
        std = self.gain / math.sqrt(fan_in)
        tensor.data = normal(tensor.shape, 0, std, tensor.dtype, tensor.device).data


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

    def __call__(self, tensor: Tensor | Parameter) -> None:
        fan_in = get_fan_in(tensor.shape)
        k = self.gain * math.sqrt(3 / fan_in)
        tensor.data = uniform(tensor.shape, -k, k, tensor.dtype, tensor.device).data


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

    def __call__(self, tensor: Tensor | Parameter) -> None:
        fan_in = get_fan_in(tensor.shape)
        fan_out = _get_fan_out(tensor.shape)
        std = self.gain * math.sqrt(2 / (fan_in + fan_out))
        tensor.data = normal(tensor.shape, 0, std, tensor.dtype, tensor.device).data


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

    def __call__(self, tensor: Tensor | Parameter) -> None:
        fan_in = get_fan_in(tensor.shape)
        fan_out = _get_fan_out(tensor.shape)
        k = self.gain * math.sqrt(6 / (fan_in + fan_out))
        tensor.data = uniform(tensor.shape, -k, k, tensor.dtype, tensor.device).data


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


def get_fan_in(shape: _ShapeLike) -> int:
    """Returns the fan-in value for a given shape."""
    return reduce(operator.mul, (shape[0],) + shape[2:])


def _get_fan_out(shape: _ShapeLike) -> int:
    """Returns the fan-out value for a given shape."""
    return reduce(operator.mul, (shape[1],) + shape[2:])
