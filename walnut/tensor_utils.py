"""Tensor utils module"""

import numpy as np
from walnut.cuda import get_engine
from walnut.tensor import Tensor, ShapeError, ShapeLike, AxisLike
from walnut.cuda import ScalarLike


__all__ = [
    "expand_dims",
    "match_dims",
    "arange",
    "linspace",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    "randn",
    "randu",
    "randint",
    "random_permutation",
    "shuffle",
    "check_dims",
    "random_choice",
    "random_choice_indices",
    "empty",
    "maximum",
    "concatenate",
    "prod",
    "eye",
    "split",
    "unique",
]


def expand_dims(x: Tensor, axis: AxisLike) -> Tensor:
    """Extends the dimensions of a tensor.

    Parameters
    ----------
    x : Tensor
        Tensor whose dimensions are to be extended.
    axis : AxisLike
        Where to insert the new dimension.

    Returns
    -------
    Tensor
        Tensor with extended dimensions.
    """
    engine = get_engine(x.device)
    r = engine.expand_dims(x.data, axis=axis)
    return Tensor(r, dtype=r.dtype, device=x.device)


def match_dims(x: Tensor, dims: int) -> Tensor:
    """Extends the dimensions of a tensor to fit a given number of dims.

    Parameters
    ----------
    x : Tensor
        Tensor to be extended.
    dims : int
        Number of dimensions needed.

    Returns
    -------
    Tensor
        Tensor with extended dimensions.
    """
    while x.ndim < dims:
        x = expand_dims(x, axis=(-1,))
    return x


def arange(
    stop: int,
    start: int = 0,
    step: int | float = 1,
    dtype: str = "int32",
    device: str = "cpu",
) -> Tensor:
    """Returns a 1d tensor with evenly spaced values samples,
    calculated over the interval [start, stop).

    Parameters
    ----------
    start : float
        Start value.
    stop : float
        Stop value.
    step : int | float, optional
        Spacing between values, by default 1.
    dtype: str, optional
        Datatype of the tensor data, by default "int32".
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        1d tensor of evenly spaced samples.
    """
    engine = get_engine(device)
    r = engine.arange(start, stop, step, dtype=dtype)
    return Tensor(r, dtype=r.dtype, device=device)


def linspace(
    start: float, stop: float, num: int, dtype: str = "float64", device: str = "cpu"
) -> Tensor:
    """Returns a 1d tensor num evenly spaced samples, calculated over the interval [start, stop].

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
        1d tensor of evenly spaced samples.
    """
    engine = get_engine(device)
    r = engine.linspace(start, stop, num, dtype=dtype)
    return Tensor(r, dtype=r.dtype, device=device)


def zeros(
    shape: ShapeLike,
    dtype: str = "float64",
    device: str = "cpu",
) -> Tensor:
    """Creates a tensor of a given shape with all values being zero.

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
    engine = get_engine(device)
    r = engine.zeros(shape, dtype=dtype)
    return Tensor(r, dtype=r.dtype, device=device)


def ones(shape: ShapeLike, dtype: str = "float64", device: str = "cpu") -> Tensor:
    """Creates a tensor of a given shape with all values being one.

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
    engine = get_engine(device)
    r = engine.ones(shape, dtype=dtype)
    return Tensor(r, dtype=r.dtype, device=device)


def zeros_like(x: Tensor, dtype: str = "float64", device: str = "cpu") -> Tensor:
    """Creates a tensor based on the shape of a given other tensor with all values being zero.

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
    """Creates a tensor based on the shape of a given other tensor with all values being one.

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


def randn(
    shape: ShapeLike,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: str = "float64",
    device: str = "cpu",
) -> Tensor:
    """Creates a tensor of a given shape with random values following a normal distribution.

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
        Tensor with random values.
    """
    engine = get_engine(device)
    r = engine.random.normal(mean, std, shape)
    return Tensor(r, dtype=dtype, device=device)


def randu(
    shape: ShapeLike,
    low: float = 0.0,
    high: float = 1.0,
    dtype: str = "float64",
    device: str = "cpu",
) -> Tensor:
    """Creates a tensor of a given shape with random values following a uniform distribution.

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
        Tensor with random values.
    """
    engine = get_engine(device)
    r = engine.random.uniform(low, high, shape)
    return Tensor(r, dtype=dtype, device=device)


def randint(
    shape: ShapeLike, low: int, high: int, dtype: str = "int32", device: str = "cpu"
) -> Tensor:
    """Creates a tensor of a given shape with random integer values.

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
        Tensor with random values.
    """
    engine = get_engine(device)
    r = engine.random.randint(low, high, shape)
    return Tensor(r, dtype=dtype, device=device)


def random_permutation(n: int, dtype: str = "int32", device: str = "cpu") -> Tensor:
    """Returns a permuted range of length n.

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
        Permuted range tensor.
    """
    engine = get_engine(device)
    r = engine.random.permutation(n)
    return Tensor(r, dtype=dtype, device=device)


def shuffle(x: Tensor) -> tuple[Tensor, Tensor]:
    """Shuffles a tensor along axis 0.

    Parameters
    ----------
    x : Tensor
        First tensor to be shuffled.

    Returns
    -------
    tuple[Tensor, Tensor]
        Shuffled tensor and index tensor.
    """
    shuffle_idx = random_permutation(x.len, device=x.device)
    return x[shuffle_idx], shuffle_idx


def check_dims(x: Tensor, target_dim: int) -> None:
    """Checks if a tensors dimensions match desired target dimensions.

    Parameters
    ----------
    x : Tensor
        Tensor whose dimensions are checked.
    target_dim : int
        Number of dimension the tensor should have.

    Raises
    ------
    ShapeError
        If the tensor's dimensions do not match the target dimensions.
    """
    if x.ndim != target_dim:
        raise ShapeError("Input dimensions do not match.")


def random_choice_indices(p: Tensor, num_samples: int = 1) -> Tensor:
    """Returns random indices based on a probability distribution tensor.

    Parameters
    ----------
    p : Tensor
        Probablitity distribution.
    num_samples : int, optional
        Number of samples to draw, by default 1.

    Returns
    -------
    Tensor
        Chosen samples.
    """
    engine = get_engine(p.device)
    return Tensor(
        engine.random.choice(p.len, size=num_samples, p=p.data),
        dtype="int32",
        device=p.device,
    )


def random_choice(
    x: Tensor,
    p: Tensor,
    shape: ShapeLike,
    device: str = "cpu",
) -> Tensor:
    """_summary_

    Parameters
    ----------
    x : Tensor
        Tensor of possible values.
    p : Tensor
        Probablitity distribution.
    shape : ShapeLike
        Shape of the random tensor.
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor of random choices.
    """
    engine = get_engine(device)
    r = engine.random.choice(x.data, shape, p=p.data)
    return Tensor(r, dtype=r.dtype, device=device)


def empty(dtype: str = "float64", device: str = "cpu") -> Tensor:
    """Return an empty tensor.

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
    # does nto accept string?
    engine = get_engine(device)
    r = engine.empty(0, dtype=dtype)
    return Tensor(r, dtype=r.dtype, device=device)


def maximum(a: Tensor | ScalarLike, b: Tensor | ScalarLike) -> Tensor:
    """Element-wise maximum of two tensors or values.

    Parameters
    ----------
    a : Tensor | float | int
        First value.
    b : Tensor | float | int
        Second value.

    Returns
    -------
    Tensor
        Tensor containing the element-wise maximum of either tensor.
    """
    device = "cpu"

    if isinstance(a, Tensor):
        _a = a.data
        device = a.device
    else:
        _a = a

    if isinstance(b, Tensor):
        _b = b.data
        device = b.device
    else:
        _b = b

    engine = get_engine(device)
    r = engine.maximum(_a, _b)
    return Tensor(r, dtype=r.dtype, device=device)


def concatenate(tensors: list[Tensor], axis: int = -1) -> Tensor:
    """Joins a sequence of tensors along a given axis.

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
    engine = get_engine(device)
    r = engine.concatenate([t.data for t in tensors], axis=axis)
    return Tensor(r, dtype=r.dtype, device=device)


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
    """Returns a diagonal tensor of size n x n.

    Parameters
    ----------
    size : int
        Size of the tensor
    dtype: str, optional
        Datatype of the tensor data, by default float64.
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Diagonal tensor.
    """
    engine = get_engine(device)
    r = engine.eye(n, dtype=dtype)
    return Tensor(r, dtype=r.dtype, device=device)


def split(x: Tensor, splits: int | list[int], axis: int = -1) -> list[Tensor]:
    """Splits a tensor into a list of sub-tensors.

    Parameters
    ----------
    x : Tensor
        Tensor to split.
    splits : int | list[int]
        If an int is given, the tensor is split into n equally sized tensors.
        If a list of indices is given, they represent the indices at which to
        split the tensor along the given axis.
    axis : int, optional
        Axis along which to split the tensor, by default -1

    Returns
    -------
    list[Tensor]
        List of tensors containing the split data.
    """
    engine = get_engine(x.device)
    split_data = engine.split(x.data, splits, axis=axis)
    return [Tensor(s, dtype=x.dtype, device=x.device) for s in split_data]


def unique(x: Tensor) -> Tensor:
    """Returns a tensor of unique values.

    Parameters
    ----------
    x : Tensor
        Tensor.

    Returns
    -------
    Tensor
        Tensor containing the unique ordered values.
    """
    engine = get_engine(x.device)
    return Tensor(engine.unique(x.data), dtype=x.dtype, device=x.device)
