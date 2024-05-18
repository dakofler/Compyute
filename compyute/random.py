"""Random functions module"""

from ._types import _DeviceLike, _DtypeLike, _ShapeLike
from .engine import _get_engine
from .engine import gpu_available as _gpu_available
from .tensors import Tensor, tensor

__all__ = [
    "normal",
    "uniform",
    "uniform_int",
    "permutation",
    "set_seed",
    "shuffle",
    "multinomial",
]


def set_seed(seed: int) -> None:
    """Sets the seed for RNG for reproducability.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    if _gpu_available():
        _get_engine("cuda").random.seed(seed)
    _get_engine("cpu").random.seed(seed)


def normal(
    shape: _ShapeLike,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: _DtypeLike = "float64",
    device: _DeviceLike = "cpu",
) -> Tensor:
    """Returns a tensor with values drawn from a normal distribution.

    Parameters
    ----------
    ShapeLike
        Shape of the new tensor.
    mean : float, optional
        Mean of random values, by default 0.
    std : float, optional
        Standard deviation of random values, by default 1.
    dtype: str, optional
        Datatype of the tensor data, by default "float64".
    device: str, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor of normally distributed samples.
    """
    return tensor(_get_engine(device).random.normal(mean, std, shape), device=device, dtype=dtype)


def uniform(
    shape: _ShapeLike,
    low: float = -1.0,
    high: float = 1.0,
    dtype: _DtypeLike = "float64",
    device: _DeviceLike = "cpu",
) -> Tensor:
    """Returns a tensor with values drawn from a uniform distribution.

    Parameters
    ----------
    ShapeLike
        Shape of the new tensor.
    low : float, optional
        Lower bound for random values, by default 0.
    high : float, optional
        Upper bound for random values, by default 1.
    dtype: str, optional
        Datatype of the tensor data, by default "float64".
    device: str, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor of uniformly distributed samples.
    """
    return tensor(_get_engine(device).random.uniform(low, high, shape), device=device, dtype=dtype)


def uniform_int(
    shape: _ShapeLike,
    low: int,
    high: int,
    dtype: _DtypeLike = "int32",
    device: _DeviceLike = "cpu",
) -> Tensor:
    """Returns a tensor with integer values drawn from a discrete uniform distribution.

    Parameters
    ----------
    ShapeLike
        Shape of the new tensor.
    low : int
        Lower bound for random values.
    high : int
        Upper bound for random values.
    dtype: str, optional
        Datatype of the tensor data, by default "int32".
    device: str, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor of samples.
    """
    return tensor(_get_engine(device).random.randint(low, high, shape), device=device, dtype=dtype)


def permutation(n: int, dtype: _DtypeLike = "int32", device: _DeviceLike = "cpu") -> Tensor:
    """Returns a tensor containing a permuted range of length n.

    Parameters
    ----------
    n : int
        Length of the permuted range.
    dtype: str, optional
        Datatype of the tensor data, by default "int32".
    device: str, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Permuted tensor.
    """
    return tensor(_get_engine(device).random.permutation(n), device=device, dtype=dtype)


def multinomial(x: Tensor | int, p: Tensor, shape: _ShapeLike) -> Tensor:
    """Returns a tensor of values drawn from a given probability distribution tensor.

    Parameters
    ----------
    x : Tensor | int
        If a tensor, it represents possible values to draw.
        If an int, values are drawn from arange(x).
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
        return Tensor(_get_engine(p.device).random.choice(x, size=shape, p=p.data))
    return Tensor(_get_engine(p.device).random.choice(x.data, size=shape, p=p.data))


def multinulli(p: float, shape: _ShapeLike, device: _DeviceLike = "cpu") -> Tensor:
    """Returns a tensor of repeated bernoulli experiments using a given probability.

    Parameters
    ----------
    p : float
        Probability of success.
    shape : ShapeLike
        Shape of the new tensor.
    device: str, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor of samples.
    """
    return Tensor(_get_engine(device).random.choice([0, 1], size=shape, p=[p, 1 - p]))


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
