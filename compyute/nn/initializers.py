"""Parameter initializations module"""

import operator
from abc import ABC, abstractmethod
from functools import reduce
from typing import Literal, Optional

from .. import random
from ..base_tensor import Tensor, _ShapeLike
from ..dtypes import Dtype, _DtypeLike
from ..tensor_functions.creating import ones, zeros

__all__ = [
    "get_initializer",
    "KaimingNormal",
    "KaimingUniform",
    "Normal",
    "Uniform",
    "XavierNormal",
    "XavierUniform",
]


class Initializer(ABC):
    """Optimizer base class"""

    __slots__ = ("dtype",)

    def __init__(self, dtype: _DtypeLike) -> None:
        self.dtype = Dtype(dtype)

    @abstractmethod
    def __call__(self, shape: _ShapeLike) -> Tensor:
        """Returns a tensor with values.

        Parameters
        ----------
        shape : ShapeLike
            Shape of the new tensor.

        Returns
        -------
        Tensor
            Tensor with random values.
        """


class KaimingNormal(Initializer):
    """Normal initializer following He et al., 2015."""

    __slots__ = ("gain",)

    def __init__(self, gain: float = 1.0, dtype: _DtypeLike = Dtype.FLOAT32) -> None:
        """Normal initializer following He et al., 2015.

        Parameters
        ----------
        gain : float, optional
            Gain value to use for initializing values.
        dtype: _DtypeLike, optional
            Datatype of the tensor data, by default Dtype.FLOAT32.
        """
        super().__init__(dtype)
        self.gain = gain

    def __call__(self, shape: _ShapeLike) -> Tensor:
        fan_in = _get_fan_in(shape)
        std = self.gain / fan_in**0.5
        return random.normal(shape, std=std, dtype=self.dtype)


class KaimingUniform(Initializer):
    """Uniform initializer following He et al., 2015."""

    __slots__ = ("gain",)

    def __init__(self, gain: float = 1.0, dtype: _DtypeLike = Dtype.FLOAT32) -> None:
        """Uniform initializer following He et al., 2015.

        Parameters
        ----------
        gain : float, optional
            Gain value to use for initializing values.
        dtype: _DtypeLike, optional
            Datatype of the tensor data, by default Dtype.FLOAT32.
        """
        super().__init__(dtype)
        self.gain = gain

    def __call__(self, shape: _ShapeLike) -> Tensor:
        fan_in = _get_fan_in(shape)
        k = self.gain * (3 / fan_in) ** 0.5
        return random.uniform(shape, low=-k, high=k, dtype=self.dtype)


class Normal(Initializer):
    """Initializes a tensor with values following a normal distribution."""

    __slots__ = ("mean", "std")

    def __init__(
        self, mean: float = 0.0, std: float = 1.0, dtype: _DtypeLike = Dtype.FLOAT32, **kwargs
    ) -> None:
        """Initializes a tensor with values following a uniform distribution.

        Parameters
        ----------
        mean : float, optional
            Mean of the normal distribution, by default 0.
        std : float, optional
            Standard deviation of the normal distribution, by default 1.
        dtype: _DtypeLike, optional
            Datatype of the tensor data, by default Dtype.FLOAT32.
        """
        super().__init__(dtype)
        self.mean = mean
        self.std = std

    def __call__(self, shape: _ShapeLike) -> Tensor:
        return random.normal(shape, self.mean, self.std, self.dtype)


class Uniform(Initializer):
    """Initializes a tensor with values following a uniform distribution."""

    __slots__ = ("low", "high")

    def __init__(
        self, low: float = 0.0, high: float = 1.0, dtype: _DtypeLike = Dtype.FLOAT32, **kwargs
    ) -> None:
        """Initializes a tensor with values following a uniform distribution.

        Parameters
        ----------
        low : float, optional
            Lower bound for random values, by default 0.
        high : float, optional
            Upper bound for random values, by default 1.
        dtype: _DtypeLike, optional
            Datatype of the tensor data, by default Dtype.FLOAT32.
        """
        super().__init__(dtype)
        self.low = low
        self.high = high

    def __call__(self, shape: _ShapeLike) -> Tensor:
        return random.uniform(shape, self.low, self.high, self.dtype)


class XavierNormal(Initializer):
    """Normal initializer following Glorot et al., 2010."""

    __slots__ = ("gain",)

    def __init__(self, gain: float = 1.0, dtype: _DtypeLike = Dtype.FLOAT32) -> None:
        """Normal initializer following Glorot et al., 2010.

        Parameters
        ----------
        gain : float, optional
            Gain value to use for initializing values, by default 1.
        dtype: _DtypeLike, optional
            Datatype of the tensor data, by default Dtype.FLOAT32.
        """
        super().__init__(dtype)
        self.gain = gain

    def __call__(self, shape: _ShapeLike) -> Tensor:
        fan_in = _get_fan_in(shape)
        fan_out = _get_fan_out(shape)
        std = self.gain * (2 / (fan_in + fan_out)) ** 0.5
        return random.normal(shape, std=std, dtype=self.dtype)


class XavierUniform(Initializer):
    """Uniform initializer following Glorot et al., 2010."""

    __slots__ = ("gain",)

    def __init__(self, gain: float = 1.0, dtype: _DtypeLike = Dtype.FLOAT32) -> None:
        """Uniform initializer following Glorot et al., 2010.

        Parameters
        ----------
        gain : float, optional
            Gain value to use for initializing values, by default 1.
        dtype: _DtypeLike, optional
            Datatype of the tensor data, by default Dtype.FLOAT32.
        """
        super().__init__(dtype)
        self.gain = gain

    def __call__(self, shape: _ShapeLike) -> Tensor:
        fan_in = _get_fan_in(shape)
        fan_out = _get_fan_out(shape)
        k = self.gain * (6 / (fan_in + fan_out)) ** 0.5
        return random.uniform(shape, low=-k, high=k, dtype=self.dtype)


class Ones(Initializer):
    """Initializes a tensor with ones."""

    __slots__ = ()

    def __init__(self, dtype: _DtypeLike = Dtype.FLOAT32, **kwargs) -> None:
        """Initializes a tensor with ones.

        Parameters
        ----------
        dtype: _DtypeLike, optional
            Datatype of the tensor data, by default Dtype.FLOAT32.
        """
        super().__init__(dtype)

    def __call__(self, shape: _ShapeLike) -> Tensor:
        return ones(shape, self.dtype)


class Zeros(Initializer):
    """Initializes a tensor with zeros."""

    __slots__ = ()

    def __init__(self, dtype: _DtypeLike = Dtype.FLOAT32, **kwargs) -> None:
        """Initializes a tensor with zeros.

        Parameters
        ----------
        dtype: _DtypeLike, optional
            Datatype of the tensor data, by default Dtype.FLOAT32.
        """
        super().__init__(dtype)

    def __call__(self, shape: _ShapeLike) -> Tensor:
        return zeros(shape, self.dtype)


def get_initializer(
    initializer: str,
    dtype: _DtypeLike,
    activation_function: Optional[Literal["relu", "leaky_relu", "gelu", "sigmoid", "tanh"]] = None,
) -> Initializer:
    """Returns an instance of an initializer."""
    if isinstance(initializer, Initializer):
        return initializer

    initializers = {
        "kaiming_normal": KaimingNormal,
        "kaiming_uniform": KaimingUniform,
        "normal": Normal,
        "uniform": Uniform,
        "xavier_normal": XavierNormal,
        "xavier_uniform": XavierUniform,
        "zeros": Zeros,
        "ones": Ones,
    }

    if initializer not in initializers.keys():
        raise ValueError(f"Unknown initializer {initializer}.")

    gains = {
        "sigmoid": 1,
        "tanh": 5 / 3,
        "relu": 2**0.5,
        "leaky_relu": (2 / (1 + 0.1**2)) ** 0.5,
        "selu": 3 / 4,
    }
    if activation_function is not None:
        gain = gains.get(activation_function, 1)
    else:
        gain = 1

    return initializers[initializer](dtype=dtype, gain=gain)


def _get_fan_in(shape: _ShapeLike) -> int:
    return reduce(operator.mul, (shape[0],) + shape[2:])


def _get_fan_out(shape: _ShapeLike) -> int:
    return reduce(operator.mul, (shape[1],) + shape[2:])
