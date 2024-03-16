"""Tensor utils module"""

import numpy as np
from compyute.engine import get_engine
from compyute.tensor import Tensor
from compyute.engine import DeviceLike, DtypeLike, ScalarLike, ShapeLike


__all__ = [
    "arange",
    "linspace",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    "random_normal",
    "random_uniform",
    "random_int",
    "random_permutation",
    "shuffle",
    "random_multinomial",
    "random_multinomial_idx",
    "empty",
    "maximum",
    "concatenate",
    "prod",
    "eye"
]


def arange(
    stop: int,
    start: int = 0,
    step: int | float = 1,
    dtype: DtypeLike = "int32",
    device: DeviceLike = "cpu",
) -> Tensor:
    """Returns a tensor of evenly spaced values using a step size within
    a given interval [start, stop).

    Parameters
    ----------
    stop : float
        Stop value (not included).
    start : float, optional
        Start value, by default 0.
    step : int | float, optional
        Spacing between values, by default 1.
    dtype: str, optional
        Datatype of the tensor data, by default "int32".
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor of evenly spaced samples.
    """
    return Tensor(get_engine(device).arange(start, stop, step, dtype=dtype))


def linspace(
    start: float,
    stop: float,
    num: int,
    dtype: DtypeLike = "float64",
    device: DeviceLike = "cpu",
) -> Tensor:
    """Returns a tensor of num evenly spaced values within
    a given interval [start, stop].

    Parameters
    ----------
    start : float
        Start value.
    stop : float
        Stop value.
    num : int
        Number of samples.
    dtype: str, optional
        Datatype of the tensor data, by default "float64".
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor of evenly spaced samples.
    """
    return Tensor(get_engine(device).linspace(start, stop, num, dtype=dtype))


def zeros(
    shape: ShapeLike,
    dtype: DtypeLike = "float64",
    device: DeviceLike = "cpu",
) -> Tensor:
    """Returns a tensor of a given shape with all values being zero.

    Parameters
    ----------
    ShapeLike
        Shape of the new tensor.
    dtype: str, optional
        Datatype of the tensor data, by default "float64".
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with all values being zero.
    """
    return Tensor(get_engine(device).zeros(shape, dtype=dtype))


def ones(
    shape: ShapeLike, dtype: DtypeLike = "float64", device: DeviceLike = "cpu"
) -> Tensor:
    """Returns a tensor of a given shape with all values being one.

    Parameters
    ----------
    ShapeLike
        Shape of the new tensor.
    dtype: str, optional
        Datatype of the tensor data, by default "float64".
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    return Tensor(get_engine(device).ones(shape, dtype=dtype))


def zeros_like(
    x: Tensor, dtype: DtypeLike = "float64", device: DeviceLike = "cpu"
) -> Tensor:
    """Returns a tensor based on the shape of a given other tensor with all values being zero.

    Parameters
    ----------
    x : Tensor
        Tensor whose shape is used.
    dtype: str, optional
        Datatype of the tensor data, by default "float64".
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with all values being zero.
    """
    return zeros(x.shape, dtype=dtype, device=device)


def ones_like(
    x: Tensor, dtype: DtypeLike = "float64", device: DeviceLike = "cpu"
) -> Tensor:
    """Returns a tensor based on the shape of a given other tensor with all values being one.

    Parameters
    ----------
    x : Tensor
        Tensor whose shape is used.
    dtype: str, optional
        Datatype of the tensor data, by default "float64".
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    return ones(x.shape, dtype=dtype, device=device)


def random_normal(
    shape: ShapeLike,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: DtypeLike = "float64",
    device: DeviceLike = "cpu",
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
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor of normally distributed samples.
    """
    return Tensor(get_engine(device).random.normal(mean, std, shape), dtype=dtype)


def random_uniform(
    shape: ShapeLike,
    low: float = -1.0,
    high: float = 1.0,
    dtype: DtypeLike = "float64",
    device: DeviceLike = "cpu",
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
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor of uniformly distributed samples.
    """
    return Tensor(get_engine(device).random.uniform(low, high, shape), dtype=dtype)


def random_int(
    shape: ShapeLike,
    low: int,
    high: int,
    dtype: DtypeLike = "int32",
    device: DeviceLike = "cpu",
) -> Tensor:
    """Returns a tensor with values drawn from a discrete uniform distribution.

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
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor of samples.
    """
    return Tensor(get_engine(device).random.randint(low, high, shape), dtype=dtype)


def random_permutation(
    n: int, dtype: DtypeLike = "int32", device: DeviceLike = "cpu"
) -> Tensor:
    """Returns a tensor containing a permuted range of length n.

    Parameters
    ----------
    n : int
        Length of the permuted range.
    dtype: str, optional
        Datatype of the tensor data, by default "int32".
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Permuted tensor.
    """
    return Tensor(get_engine(device).random.permutation(n), dtype=dtype)


def random_multinomial_idx(p: Tensor, num_samples: int = 1) -> Tensor:
    """Returns a tensor of indices drawn from a probability distribution tensor.

    Parameters
    ----------
    p : Tensor
        Probablitity distribution.
    num_samples : int, optional
        Number of samples to draw, by default 1.

    Returns
    -------
    Tensor
        Tensor of samples.
    """
    return Tensor(
        get_engine(p.device).random.choice(
            p.shape[0], size=num_samples, p=p.data),
        dtype="int32",
    )


def random_multinomial(
    x: Tensor,
    p: Tensor,
    shape: ShapeLike,
) -> Tensor:
    """Returns a tensor of values drawn from a probability distribution tensor.

    Parameters
    ----------
    x : Tensor
        Possible values.
    p : Tensor
        Corresponding probablitity distribution.
    shape : ShapeLike
        Shape of the new tensor.

    Returns
    -------
    Tensor
        Tensor of samples.
    """
    return Tensor(get_engine(x.device).random.choice(x.data, shape, p=p.data))


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
    shuffle_idx = random_permutation(x.shape[0], device=x.device)
    return x[shuffle_idx], shuffle_idx


def empty(dtype: DtypeLike = "float64", device: DeviceLike = "cpu") -> Tensor:
    """Returns an empty tensor.

    Parameters
    ----------
    dtype: str, optional
        Datatype of the tensor data, by default float64.
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Empty tensor.
    """
    return Tensor(get_engine(device).empty(0, dtype=dtype))


def maximum(a: Tensor | ScalarLike, b: Tensor | ScalarLike) -> Tensor:
    """Returns a new tensor containing the element-wise maximum of two tensors/scalars.

    Parameters
    ----------
    a : Tensor | ScalarLike
        Tensor or scalar.
    b : Tensor | ScalarLike
        Tensor or scalar.

    Returns
    -------
    Tensor
        Tensor containing the element-wise maxima.
    """
    device = "cpu"  # set cpu in case of two scalars

    if isinstance(a, Tensor):
        device = a.device
        _a = a.data
    else:
        _a = a

    if isinstance(b, Tensor):
        device = b.device
        _b = b.data
    else:
        _b = b

    return Tensor(get_engine(device).maximum(_a, _b))


def minimum(a: Tensor | ScalarLike, b: Tensor | ScalarLike) -> Tensor:
    """Returns a new tensor containing the element-wise minimum of two tensors/scalars.

    Parameters
    ----------
    a : Tensor | ScalarLike
        Tensor or scalar.
    b : Tensor | ScalarLike
        Tensor or scalar.

    Returns
    -------
    Tensor
        Tensor containing the element-wise minima.
    """
    device = "cpu"

    if isinstance(a, Tensor):
        device = a.device
        _a = a.data
    else:
        _a = a

    if isinstance(b, Tensor):
        device = b.device
        _b = b.data
    else:
        _b = b

    return Tensor(get_engine(device).minimum(_a, _b))


def concatenate(tensors: list[Tensor], axis: int = -1) -> Tensor:
    """Returns a new tensor by joins a sequence of tensors along a given axis.

    Parameters
    ----------
    tensors : list[Tensor]
        List of Tensors to be joined.
    axis : int, optional
        Axis along which to join the tensors, by default -1.

    Returns
    -------
    Tensor
        Concatenated tensor.
    """

    device = tensors[0].device
    return Tensor(get_engine(device).concatenate([t.data for t in tensors], axis=axis))


def prod(x: tuple[int, ...]) -> int:
    """Returns the product of tuple elements.

    Parameters
    ----------
    x : tuple[int, ...]
        Tuple of integers.

    Returns
    -------
    int
        Product of tuple elements.
    """
    return np.prod(x).item()


def eye(n: int, dtype: DtypeLike = "float64", device: DeviceLike = "cpu") -> Tensor:
    """Returns a diagonal tensor of shape (n, n).

    Parameters
    ----------
    n: int
        Size of the new tensor. The shape will be (n, n).
    dtype: str, optional
        Datatype of the tensor data, by default float64.
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Diagonal tensor.
    """
    return Tensor(get_engine(device).eye(n, dtype=dtype))
