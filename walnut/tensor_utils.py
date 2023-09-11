"""Tensor utils module"""

import pandas as pd
import numpy as np
import cupy as cp
from walnut.tensor import Tensor, ShapeLike, AxisLike, ShapeError, PyTypeLike


__all__ = [
    "df_to_tensor",
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
]


def df_to_tensor(df: pd.DataFrame) -> Tensor:
    """Converts a Pandas DataFrame into a Tensor.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame object to convert.

    Returns
    -------
    Tensor
        Tensor object.
    """
    return Tensor(df.to_numpy())


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
    if x.device == "cpu":
        exp = np.expand_dims(x.data, axis=axis)
    else:
        exp = cp.expand_dims(x.data, axis=axis)
    return Tensor(exp, dtype=x.dtype, device=x.device)


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
    stop: int, start: int = 0, step: int | float = 1, device: str = "cpu"
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
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        1d tensor of evenly spaced samples.
    """
    if device == "cpu":
        x = np.arange(start, stop, step)
    else:
        x = cp.arange(start, stop, step)  # only takes ints as step?
    return Tensor(x, dtype=str(x.dtype), device=device)


def linspace(start: float, stop: float, num: int, device: str = "cpu") -> Tensor:
    """Returns a 1d tensor num evenly spaced samples, calculated over the interval [start, stop].

    Parameters
    ----------
    start : float
        Start value.
    stop : float
        Stop value.
    num : int
        Number of samples.
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        1d tensor of evenly spaced samples.
    """
    if device == "cpu":
        lin = np.linspace(start, stop, num)
    else:
        lin = cp.linspace(start, stop, num)
    return Tensor(lin, device=device)


def zeros(shape: ShapeLike, device: str = "cpu") -> Tensor:
    """Creates a tensor of a given shape with all values being zero.

    Parameters
    ----------
    ShapeLike
        Shape of the new tensor.
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with all values being zero.
    """
    z = np.zeros(shape) if device == "cpu" else cp.zeros(shape)
    return Tensor(z, device=device)


def ones(shape: ShapeLike, device: str = "cpu") -> Tensor:
    """Creates a tensor of a given shape with all values being one.

    Parameters
    ----------
    ShapeLike
        Shape of the new tensor.
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    o = np.ones(shape) if device == "cpu" else cp.ones(shape)
    return Tensor(o, device=device)


def zeros_like(x: Tensor, device: str = "cpu") -> Tensor:
    """Creates a tensor based on the shape of a given other tensor with all values being zero.

    Parameters
    ----------
    x : Tensor
        Tensor whose shape is used.
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with all values being zero.
    """
    return zeros(x.shape, device=device)


def ones_like(x: Tensor, device: str = "cpu") -> Tensor:
    """Creates a tensor based on the shape of a given other tensor with all values being one.

    Parameters
    ----------
    x : Tensor
        Tensor whose shape is used.
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    return ones(x.shape, device=device)


def randn(
    shape: ShapeLike, mean: float = 0.0, std: float = 1.0, device: str = "cpu"
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
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with random values.
    """
    if device == "cpu":
        rand = np.random.normal(mean, std, shape)
    else:
        rand = cp.random.normal(mean, std, shape)
    return Tensor(rand, device=device)


def randu(
    shape: ShapeLike, low: float = 0.0, high: float = 1.0, device: str = "cpu"
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
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with random values.
    """
    if device == "cpu":
        rand = np.random.uniform(low, high, shape)
    else:
        rand = cp.random.uniform(low, high, shape)
    return Tensor(rand, device=device)


def randint(shape: ShapeLike, low: int, high: int, device: str = "cpu") -> Tensor:
    """Creates a tensor of a given shape with random integer values.

    Parameters
    ----------
    ShapeLike
        Shape of the new tensor.
    low : int
        Lower bound for random values.
    high : int
        Upper bound for random values.
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor with random values.
    """
    if device == "cpu":
        rand = np.random.randint(low, high, shape)
    else:
        rand = cp.random.randint(low, high, shape)
    return Tensor(rand, dtype="int", device=device)


def random_permutation(n: int, device: str = "cpu") -> Tensor:
    """Returns a permuted range of length n.

    Parameters
    ----------
    n : int
        Length of the permuted range.
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Permuted range tensor.
    """
    rand = np.random.permutation(n) if device == "cpu" else cp.random.permutation(n)
    return Tensor(rand, dtype="int", device=device)


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
    shuffle_idx = random_permutation(x.len, x.device)
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
    if p.device == "cpu":
        rand = np.random.choice(p.len, size=num_samples, p=p.data)
    else:
        rand = cp.random.choice(p.len, size=num_samples, p=p.data)
    return Tensor(rand, dtype="int", device=p.device)


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
    if device == "cpu":
        rand = np.random.choice(x.data, shape, p=p.data)
    else:
        rand = cp.random.choice(x.data, shape, p=p.data)
    return Tensor(rand, dtype=x.dtype, device=device)


def empty(dtype: str = "float32", device: str = "cpu") -> Tensor:
    """Return an empty tensor.

    Parameters
    ----------
    dtype: str, optional
        Datatype of the tensor data, by default float32.
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Empty tensor.
    """
    # does nto accept string?
    emp = np.empty(0, dtype=dtype) if device == "cpu" else cp.empty(0, dtype=dtype)
    return Tensor(emp, dtype=emp.dtype, device=device)


def maximum(a: Tensor | PyTypeLike, b: Tensor | PyTypeLike) -> Tensor:
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

    m = np.maximum(_a, _b) if device == "cpu" else cp.maximum(_a, _b)
    return Tensor(m, device=device)


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
    if device == "cpu":
        cat = np.concatenate([t.data for t in tensors], axis=axis)
    else:
        cat = cp.concatenate([t.data for t in tensors], axis=axis)
    return Tensor(cat, device=device, dtype=tensors[0].dtype)


def prod(x: tuple[int, ...]) -> int | float:
    """Returns the product of tuple elements.

    Parameters
    ----------
    x : tuple[int, ...]
        _description_

    Returns
    -------
    int | float
        Product of tuple elements.
    """
    return np.prod(x).item()


def eye(n: int, device: str = "cpu") -> Tensor:
    """Returns a diagonal tensor of size n x n.

    Parameters
    ----------
    size : int
        Size of the tensor
    device: str, optinal
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Diagonal tensor.
    """
    e = np.eye(n) if device == "cpu" else cp.eye(n)
    return Tensor(e, device=device)


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
    Tensor
        _description_
    """
    if x.device == "cpu":
        split_data = np.split(x.data, splits, axis=axis)
    else:
        split_data = cp.split(x.data, splits, axis=axis)
    return [Tensor(s, dtype=x.dtype, device=x.device) for s in split_data]
