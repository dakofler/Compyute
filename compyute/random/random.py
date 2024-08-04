"""Randomness based tensor functions."""

from contextlib import contextmanager
from typing import Optional

from ..base_tensor import Tensor, _ShapeLike
from ..dtypes import _DtypeLike, select_dtype
from ..engine import Device, _DeviceLike, get_engine, select_device_or_cpu

__all__ = [
    "normal",
    "uniform",
    "uniform_int",
    "permutation",
    "seed",
    "set_seed",
    "shuffle",
    "multinomial",
]


def set_seed(value: Optional[int] = None) -> None:
    """Sets the seed of the random number generator for reproducability.

    Parameters
    ----------
    value : int, optional
        Seed value. Defaults to ``None``. If ``None``, the seed is reset.
    """
    try:
        get_engine(Device.CUDA).random.seed(value)
    except Exception:
        pass
    get_engine(Device.CPU).random.seed(value)


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
    dtype: Optional[_DtypeLike] = None,
    device: Optional[_DeviceLike] = None,
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
        Datatype of the tensor data. Defaults to ``None``.
    device : _DeviceLike, optional
        The device the tensor is stored on. Defaults to ``None``.

    Returns
    -------
    Tensor
        Tensor of normally distributed samples.
    """
    device = select_device_or_cpu(device)
    dtype = select_dtype(dtype)
    data = get_engine(device).random.normal(mean, std, shape)
    if dtype is not None:
        return Tensor(data).as_type(dtype)
    return Tensor(data)


def uniform(
    shape: _ShapeLike,
    low: float = -1.0,
    high: float = 1.0,
    dtype: Optional[_DtypeLike] = None,
    device: Optional[_DeviceLike] = None,
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
        Datatype of the tensor data. Defaults to ``None``.
    device : _DeviceLike, optional
        The device the tensor is stored on. Defaults to ``None``.

    Returns
    -------
    Tensor
        Tensor of uniformly distributed samples.
    """
    device = select_device_or_cpu(device)
    dtype = select_dtype(dtype)
    data = get_engine(device).random.uniform(low, high, shape)
    if dtype is not None:
        return Tensor(data).as_type(dtype)
    return Tensor(data)


def uniform_int(
    shape: _ShapeLike,
    low: int,
    high: int,
    dtype: Optional[_DtypeLike] = None,
    device: Optional[_DeviceLike] = None,
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
        Datatype of the tensor data. Defaults to ``None``.
    device : _DeviceLike, optional
        The device the tensor is stored on. Defaults to ``None``.

    Returns
    -------
    Tensor
        Tensor of samples.
    """
    device = select_device_or_cpu(device)
    dtype = select_dtype(dtype)
    data = get_engine(device).random.randint(low, high, shape)
    if dtype is not None:
        return Tensor(data).as_type(dtype)
    return Tensor(data)


def permutation(
    n: int, dtype: Optional[_DtypeLike] = None, device: Optional[_DeviceLike] = None
) -> Tensor:
    """Returns a tensor containing a permuted range of a specified length.

    Parameters
    ----------
    n : int
        Length of the permuted range.
    dtype : _DtypeLike, optional
        Datatype of the tensor data. Defaults to ``None``.
    device : _DeviceLike, optional
        The device the tensor is stored on. Defaults to ``None``.

    Returns
    -------
    Tensor
        Permuted tensor.
    """
    device = select_device_or_cpu(device)
    dtype = select_dtype(dtype)
    data = get_engine(device).random.permutation(n)
    if dtype is not None:
        return Tensor(data).as_type(dtype)
    return Tensor(data)


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
        return Tensor(get_engine(p.device).random.choice(x, size=shape, p=p.data))
    return Tensor(get_engine(p.device).random.choice(x.data, size=shape, p=p.data))


def multinulli(p: float, shape: _ShapeLike, device: Optional[_DeviceLike] = None) -> Tensor:
    """Returns a tensor of repeated bernoulli experiments using a given probability.

    Parameters
    ----------
    p : float
        Probability of success.
    shape : _ShapeLike
        Shape of the new tensor.
    device : _DeviceLike, optional
        The device the tensor is stored on. Defaults to ``None``.

    Returns
    -------
    Tensor
        Tensor of samples.
    """
    return Tensor(get_engine(device).random.choice([0, 1], size=shape, p=[p, 1 - p]))


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
