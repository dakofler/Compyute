"""Randomness based tensor functions."""

from contextlib import contextmanager
from typing import Optional

from ..base_tensor import Tensor, _ShapeLike, tensor
from ..dtypes import Dtype, _DtypeLike
from ..engine import Device, _DeviceLike, get_engine, gpu_available

__all__ = [
    "normal",
    "uniform",
    "uniform_int",
    "permutation",
    "seed",
    "set_seed",
    "shuffle",
    "multinomial",
    "multinulli",
]


def set_seed(value: Optional[int] = None) -> None:
    """Sets the seed of the random number generator for reproducability.

    Parameters
    ----------
    value : int, optional
        Seed value. Defaults to ``None``. If ``None``, the seed is reset.
    """
    get_engine(Device.CPU).random.seed(value)
    if gpu_available():
        get_engine(Device.CUDA).random.seed(value)


@contextmanager
def seed(value: int):
    """Context manager to set the seed of the random number generator for reproducability.

    Parameters
    ----------
    value : int
        Seed value.
    """
    set_seed(value)
    try:
        yield
    finally:
        set_seed()


def normal(
    shape: _ShapeLike,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: _DtypeLike = Dtype.FLOAT32,
    device: _DeviceLike = Device.CPU,
) -> Tensor:
    """Creates a tensor with values drawn from a normal distribution.

    Parameters
    ----------
    shape : _ShapeLike
        Shape of the new tensor.
    mean : float, optional
        Mean of random values. Defaults to ``0``.
    std : float, optional
        Standard deviation of random values. Defaults to ``1``.
    dtype : _DtypeLike, optional
        Datatype of the tensor data. Defaults to :class:`compyute.float32`.
    device : _DeviceLike, optional
        The device the tensor is stored on. Defaults to :class:`compyute.cpu`.

    Returns
    -------
    Tensor
        Tensor of normally distributed samples.
    """
    dtype = Dtype(dtype).value
    return tensor(get_engine(device).random.normal(mean, std, shape), device=device, dtype=dtype)


def uniform(
    shape: _ShapeLike,
    low: float = -1.0,
    high: float = 1.0,
    dtype: _DtypeLike = Dtype.FLOAT32,
    device: _DeviceLike = Device.CPU,
) -> Tensor:
    """Creates a tensor with values drawn from a uniform distribution.

    Parameters
    ----------
    shape : _ShapeLike
        Shape of the new tensor.
    low : float, optional
        Lower bound for random values. Defaults to ``0``.
    high : float, optional
        Upper bound for random values. Defaults to ``1``.
    dtype : _DtypeLike, optional
        Datatype of the tensor data. Defaults to :class:`compyute.float32`
    device : _DeviceLike, optional
        The device the tensor is stored on. Defaults to :class:`compyute.cpu`.

    Returns
    -------
    Tensor
        Tensor of uniformly distributed samples.
    """
    dtype = Dtype(dtype).value
    return tensor(get_engine(device).random.uniform(low, high, shape), device=device, dtype=dtype)


def uniform_int(
    shape: _ShapeLike,
    low: int,
    high: int,
    dtype: _DtypeLike = Dtype.INT64,
    device: _DeviceLike = Device.CPU,
) -> Tensor:
    """Creates a tensor with integer values drawn from a discrete uniform distribution.

    Parameters
    ----------
    shape : _ShapeLike
        Shape of the new tensor.
    low : int
        Lower bound for random values.
    high : int
        Upper bound for random values.
    dtype : _DtypeLike, optional
        Datatype of the tensor data. Defaults to :class:`compyute.float32`
    device : _DeviceLike, optional
        The device the tensor is stored on. Defaults to :class:`compyute.cpu`.

    Returns
    -------
    Tensor
        Tensor of samples.
    """
    dtype = Dtype(dtype).value
    return tensor(get_engine(device).random.randint(low, high, shape), device=device, dtype=dtype)


def permutation(n: int, dtype: _DtypeLike = Dtype.INT64, device: _DeviceLike = Device.CPU) -> Tensor:
    """Returns a tensor containing a permuted range of a specified length.

    Parameters
    ----------
    n : int
        Length of the permuted range.
    dtype : _DtypeLike, optional
        Datatype of the tensor data. Defaults to :class:`compyute.int64`
    device : _DeviceLike, optional
        The device the tensor is stored on. Defaults to :class:`compyute.cpu`.

    Returns
    -------
    Tensor
        Permuted tensor.
    """
    dtype = Dtype(dtype).value
    return tensor(get_engine(device).random.permutation(n), device=device, dtype=dtype)


def multinomial(x: Tensor | int, p: Tensor, shape: _ShapeLike) -> Tensor:
    """Returns a tensor of values drawn from a given probability distribution tensor.

    Parameters
    ----------
    x : Tensor | int
        If a tensor, it represents possible values to draw.
        If an int, values are drawn from ``arange(x)``.
    p : Tensor
        Corresponding probablitity distribution.
    shape : _ShapeLike
        Shape of the new tensor.

    Returns
    -------
    Tensor
        Tensor of samples.
    """
    if isinstance(x, int):
        return tensor(get_engine(p.device).random.choice(x, size=shape, p=p.data))
    return Tensor(get_engine(p.device).random.choice(x.data, size=shape, p=p.data))


def multinulli(p: float, shape: _ShapeLike, device: _DeviceLike = Device.CPU) -> Tensor:
    """Returns a tensor of repeated bernoulli experiments using a given probability.

    Parameters
    ----------
    p : float
        Probability of success.
    shape : _ShapeLike
        Shape of the new tensor.
    device : _DeviceLike, optional
        The device the tensor is stored on. Defaults to :class:`compyute.cpu`.

    Returns
    -------
    Tensor
        Tensor of samples.
    """
    return Tensor(get_engine(device).random.choice([0, 1], size=shape, p=[p, 1 - p]).astype("float32"))


def shuffle(x: Tensor) -> tuple[Tensor, Tensor]:
    """Shuffles a tensor along axis 0.

    Parameters
    ----------
    x : Tensor
        Tensor to be shuffled.

    Returns
    -------
    Tensor
        Shuffled tensor.
    Tensor
        Indices tensor.
    """
    shuffle_idx = permutation(x.shape[0], device=x.device)
    return x[shuffle_idx], shuffle_idx
