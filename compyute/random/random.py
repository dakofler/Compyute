"""Randomness based tensor functions."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Optional

from ..backend import Device, cpu, cuda, gpu_available, select_device
from ..tensors import ShapeLike, Tensor
from ..typing import DType, int64, select_dtype

__all__ = [
    "random",
    "normal",
    "uniform",
    "uniform_int",
    "permutation",
    "seed",
    "set_seed",
    "shuffle",
    "multinomial",
    "bernoulli",
]


def set_seed(value: Optional[int] = None) -> None:
    """Sets the seed of the random number generator for reproducability.

    Parameters
    ----------
    value : int, optional
        Seed value. Defaults to ``None``. If ``None``, the seed is reset.
    """
    cpu.module.random.seed(value)
    if gpu_available():
        cuda.module.random.seed(value)


@contextmanager
def seed(value: int) -> Generator:
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


def random(
    shape: ShapeLike,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DType] = None,
) -> Tensor:
    """Creates a tensor with random values in the half-open interval ``[0.0, 1.0)``.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the new tensor.
    device : Device, optional
        The device the tensor is stored on. Defaults to ``None``.
    dtype : DtypeLike, optional
        Datatype of the tensor data. Defaults to ``None``.

    Returns
    -------
    Tensor
        Tensor with random values.
    """
    device = select_device(device)
    dtype = select_dtype(dtype)
    with device:
        data = device.module.random.random(shape)
    return Tensor(data.astype(dtype.t, copy=False))


def normal(
    shape: ShapeLike,
    mean: float = 0.0,
    std: float = 1.0,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DType] = None,
) -> Tensor:
    """Creates a tensor with values drawn from a normal distribution.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the new tensor.
    mean : float, optional
        Mean of random values. Defaults to ``0``.
    std : float, optional
        Standard deviation of random values. Defaults to ``1``.
    device : Device, optional
        The device the tensor is stored on. Defaults to ``None``.
    dtype : DtypeLike, optional
        Datatype of the tensor data. Defaults to ``None``.

    Returns
    -------
    Tensor
        Tensor of normally distributed samples.
    """
    device = select_device(device)
    dtype = select_dtype(dtype)
    with device:
        data = device.module.random.normal(mean, std, shape)
    return Tensor(data.astype(dtype.t, copy=False))


def uniform(
    shape: ShapeLike,
    low: float = -1.0,
    high: float = 1.0,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DType] = None,
) -> Tensor:
    """Creates a tensor with values drawn from a uniform distribution.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the new tensor.
    low : float, optional
        Lower bound for random values. Defaults to ``0``.
    high : float, optional
        Upper bound for random values. Defaults to ``1``.
    device : Device, optional
        The device the tensor is stored on. Defaults to ``None``.
    dtype : DtypeLike, optional
        Datatype of the tensor data. Defaults to ``None``.

    Returns
    -------
    Tensor
        Tensor of uniformly distributed samples.
    """
    device = select_device(device)
    dtype = select_dtype(dtype)
    with device:
        data = device.module.random.uniform(low, high, shape)
    return Tensor(data.astype(dtype.t, copy=False))


def uniform_int(
    shape: ShapeLike,
    low: int,
    high: int,
    *,
    device: Optional[Device] = None,
    dtype: DType = int64,
) -> Tensor:
    """Creates a tensor with integer values drawn from a discrete uniform distribution.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the new tensor.
    low : int
        Lower bound for random values.
    high : int
        Upper bound for random values.
    device : Device, optional
        The device the tensor is stored on. Defaults to ``None``.
    dtype : DtypeLike, optional
        Datatype of the tensor data. Defaults to :class:`compyute.int64`.

    Returns
    -------
    Tensor
        Tensor of samples.
    """
    device = select_device(device)
    with device:
        data = device.module.random.randint(low, high, shape, dtype.t)
    return Tensor(data)


def permutation(n: int, *, device: Optional[Device] = None) -> Tensor:
    """Returns a tensor containing a permuted range of a specified length.

    Parameters
    ----------
    n : int
        Length of the permuted range.
    device : Device, optional
        The device the tensor is stored on. Defaults to ``None``.

    Returns
    -------
    Tensor
        Permuted tensor.
    """
    device = select_device(device)
    with device:
        data = device.module.random.permutation(n)
    return Tensor(data)


def multinomial(x: Tensor | int, p: Tensor, shape: ShapeLike) -> Tensor:
    """Returns a tensor of values drawn from a given probability distribution tensor.

    Parameters
    ----------
    x : Tensor | int
        If a tensor, it represents possible values to draw.
        If an int, values are drawn from ``arange(x)``.
    p : Tensor
        Corresponding probablitity distribution.
    shape : ShapeLike
        Shape of the new tensor.

    Returns
    -------
    Tensor
        Tensor of samples.
    """
    if isinstance(x, int):
        with p.device:
            data = p.device.module.random.choice(x, shape, p=p.data)
        return Tensor(data.astype(int64.t, copy=False))

    with p.device:
        data = p.device.module.random.choice(x.data, shape, p=p.data)
    return Tensor(data.astype(x.dtype.t, copy=False))


def bernoulli(
    p: float,
    shape: ShapeLike,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DType] = None,
) -> Tensor:
    """Returns a tensor of bernoulli experiments using a given probability.

    Parameters
    ----------
    p : float
        Probability of success.
    shape : ShapeLike
        Shape of the new tensor.
    device : Device, optional
        The device the tensor is stored on. Defaults to ``None``.
    dtype : DtypeLike, optional
        Datatype of the tensor data. Defaults to ``None``.

    Returns
    -------
    Tensor
        Tensor of samples.
    """
    device = select_device(device)
    dtype = select_dtype(dtype)
    with device:
        data = device.module.random.random(shape) < p
    return Tensor(data.astype(dtype.t, copy=False))


def shuffle(x: Tensor) -> tuple[Tensor, Tensor]:
    """Shuffles a tensor along the first dimension.

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
