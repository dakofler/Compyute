"""Tensor utils module"""

import numpy as np
from compyute.engine import get_engine
from compyute.tensor import Tensor, ShapeLike, AxisLike
from compyute.engine import ScalarLike


__all__ = [
    "insert_dim",
    "match_dims",
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
    "eye",
    "split",
    "unique",
]


def insert_dim(x: Tensor, axis: AxisLike) -> Tensor:
    """Returns a view of a tensor containing an added dimension at a given axis.

    Parameters
    ----------
    x : Tensor
        Tensor whose dimensions are to be extended.
    axis : AxisLike
        Where to insert the new dimension.

    Returns
    -------
    Tensor
        Tensor with an added dimension.
    """
    device = x.device
    x = get_engine(device).expand_dims(x.data, axis=axis)
    return Tensor(x, dtype=x.dtype, device=device)


def match_dims(x: Tensor, dims: int) -> Tensor:
    """Returns a view of a tensor with added trailing dimensions to fit a given number of dims.

    Parameters
    ----------
    x : Tensor
        Tensor to be extended.
    dims : int
        Number of dimensions needed.

    Returns
    -------
    Tensor
        Tensor with trailing dimensions.
    """
    new_shape = x.shape + (1,) * (dims - x.ndim)
    return x.reshape(new_shape)


def arange(
    stop: int,
    start: int = 0,
    step: int | float = 1,
    dtype: str = "int32",
    device: str = "cpu",
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
    x = get_engine(device).arange(start, stop, step, dtype=dtype)
    return Tensor(x, dtype=x.dtype, device=device)


def linspace(
    start: float, stop: float, num: int, dtype: str = "float64", device: str = "cpu"
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
    x = get_engine(device).linspace(start, stop, num, dtype=dtype)
    return Tensor(x, dtype=x.dtype, device=device)


def zeros(
    shape: ShapeLike,
    dtype: str = "float64",
    device: str = "cpu",
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
    x = get_engine(device).zeros(shape, dtype=dtype)
    return Tensor(x, dtype=x.dtype, device=device)


def ones(shape: ShapeLike, dtype: str = "float64", device: str = "cpu") -> Tensor:
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
    x = get_engine(device).ones(shape, dtype=dtype)
    return Tensor(x, dtype=x.dtype, device=device)


def zeros_like(x: Tensor, dtype: str = "float64", device: str = "cpu") -> Tensor:
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


def ones_like(x: Tensor, dtype: str = "float64", device: str = "cpu") -> Tensor:
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
    dtype: str = "float64",
    device: str = "cpu",
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
    x = get_engine(device).random.normal(mean, std, shape)
    return Tensor(x, dtype=dtype, device=device)


def random_uniform(
    shape: ShapeLike,
    low: float = -1.0,
    high: float = 1.0,
    dtype: str = "float64",
    device: str = "cpu",
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
    x = get_engine(device).random.uniform(low, high, shape)
    return Tensor(x, dtype=dtype, device=device)


def random_int(
    shape: ShapeLike, low: int, high: int, dtype: str = "int32", device: str = "cpu"
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
    x = get_engine(device).random.randint(low, high, shape)
    return Tensor(x, dtype=dtype, device=device)


def random_permutation(n: int, dtype: str = "int32", device: str = "cpu") -> Tensor:
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
    x = get_engine(device).random.permutation(n)
    return Tensor(x, dtype=dtype, device=device)


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
    device = p.device
    p = get_engine(device).random.choice(p.shape[0], size=num_samples, p=p.data)
    return Tensor(p, dtype="int32", device=device)


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
    device = x.device
    x = get_engine(device).random.choice(x.data, shape, p=p.data)
    return Tensor(x, dtype=x.dtype, device=device)


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


def empty(dtype: str = "float64", device: str = "cpu") -> Tensor:
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
    # does not accept string?
    x = get_engine(device).empty(0, dtype=dtype)
    return Tensor(x, dtype=x.dtype, device=device)


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

    x = get_engine(device).maximum(_a, _b)
    return Tensor(x, dtype=x.dtype, device=device)


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

    x = get_engine(device).minimum(_a, _b)
    return Tensor(x, dtype=x.dtype, device=device)


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
    x = get_engine(device).concatenate([t.data for t in tensors], axis=axis)
    return Tensor(x, dtype=x.dtype, device=device)


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


def eye(n: int, dtype: str = "float64", device: str = "cpu") -> Tensor:
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
    x = get_engine(device).eye(n, dtype=dtype)
    return Tensor(x, dtype=x.dtype, device=device)


def split(x: Tensor, splits: int | list[int], axis: int = -1) -> list[Tensor]:
    """Returns a list of tensors by splitting the original tensor.

    Parameters
    ----------
    x : Tensor
        Tensor to split.
    splits : int | list[int]
        If an int is given, the tensor is split into n equally sized tensors.
        If a list of indices is given, they represent the indices at which to
        split the tensor along the given axis.
    axis : int, optional
        Axis along which to split the tensor, by default -1.

    Returns
    -------
    list[Tensor]
        List of tensors containing the split data.
    """
    chunks = get_engine(x.device).split(x.data, splits, axis=axis)
    return [Tensor(c, dtype=x.dtype, device=x.device) for c in chunks]


def unique(x: Tensor) -> Tensor:
    """Returns a tensor of unique ordered values.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the unique ordered values.
    """
    return Tensor(get_engine(x.device).unique(x.data), dtype=x.dtype, device=x.device)
