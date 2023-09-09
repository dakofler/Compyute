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
    "shuffle",
    "check_dims",
    "choice",
    "empty",
    "maximum",
    "concatenate",
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
    return Tensor(np.expand_dims(x.data, axis=axis), dtype=x.dtype, device=x.device)


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
    x = np.arange(start, stop, step)
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
    return Tensor(np.linspace(start, stop, num), device=device)


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
    return Tensor(np.zeros(shape), device=device)


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
    return Tensor(np.ones(shape), device=device)


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
    return Tensor(np.random.normal(mean, std, shape), device=device)


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
    return Tensor(np.random.uniform(low, high, shape), device=device)


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
    return Tensor(np.random.randint(low, high, shape), dtype="int", device=device)


def shuffle(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Shuffles two tensors equally along axis 0.

    Parameters
    ----------
    x : Tensor
        First tensor to be shuffled.
    y : Tensor
        Second tensor to be shuffled.

    Returns
    -------
    tuple[Tensor, Tensor]
        Shuffled tensors.

    Raises
    ------
    ShapeError
        If tensors are not of equal size along a axis 0
    """
    if x.len != y.len:
        raise ShapeError("Tensors must have equal lengths along axis 0")

    if x.device == "cpu":
        shuffle_index = np.random.permutation(x.len)
    else:
        shuffle_index = cp.random.permutation(x.len)

    return x[shuffle_index], y[shuffle_index]


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


def choice(x: Tensor, num_samples: int = 1) -> Tensor:
    """Returns a random index based on a probability distribution tensor.

    Parameters
    ----------
    x : Tensor
        Tensor containing a probablitity distribution.
    num_samples : int, optional
        Number of samples drawn, by default 1.

    Returns
    -------
    Tensor
        Chosen samples.
    """
    if x.device == "cpu":
        arng = np.arange(x.flatten().len)
        samples = np.random.choice(arng, size=num_samples, p=x.data.flatten())
    else:
        arng = cp.arange(x.flatten().len)
        samples = cp.random.choice(arng, size=num_samples, p=x.data.flatten())

    return Tensor(samples, dtype="int", device=x.device)


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
    return Tensor(np.empty(0, dtype=dtype), device=device)


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


def stretch(
    x: Tensor,
    streching: tuple[int, int],
    target_shape: ShapeLike,
    axis: tuple[int, int] = (-2, -1),
) -> Tensor:
    """Strtches a tensor by repeating it's elements over given axis.

    Parameters
    ----------
    x : Tensor
        Tensor to be stretched out.
    streching : tuple[int, int]
        Number of repeating values along each axis.
    target_shape : ShapeLike
        Shape of the target tensor. If the shape does not match after stretching,
        remaining values are filled with zeroes.
    axis : tuple[int, int], optional
        Axis along which to stretch the tensor, by default (-2, -1).

    Returns
    -------
    Tensor
        Stretched out tensor.
    """
    fa1, fa2 = streching
    ax1, ax2 = axis

    if x.device == "cpu":
        x_stretched = np.repeat(x.data, fa1, axis=ax1)
        x_stretched = np.repeat(x_stretched, fa2, axis=ax2)
        x_resized = np.resize(x_stretched, target_shape)
    else:
        x_stretched = cp.repeat(x.data, fa1, axis=ax1)
        x_stretched = cp.repeat(x_stretched, fa2, axis=ax2)
        x_resized = cp.resize(x_stretched, target_shape)

    # resize to fit target shape by filling with zeros
    return Tensor(x_resized, device=x.device)


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

    return Tensor(cat, device=device)


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
    return Tensor(np.eye(n), device=device)
