"""Tensor utils module"""

import pandas as pd
import numpy as np
from walnut.tensor import Tensor, ShapeLike, AxisLike, ShapeError


__all__ = [
    "pd_to_tensor",
    "expand_dims",
    "match_dims",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    "randn",
    "randint",
    "shuffle",
    "check_dims",
    "choice",
    "empty",
]


def pd_to_tensor(df: pd.DataFrame) -> Tensor:
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
    return Tensor(np.expand_dims(x.data, axis=axis), dtype=x.dtype)


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


def zeros(shape: ShapeLike) -> Tensor:
    """Creates a tensor of a given shape with all values being zero.

    Parameters
    ----------
    ShapeLike
        Shape of the new tensor.

    Returns
    -------
    Tensor
        Tensor with all values being zero.
    """
    return Tensor(np.zeros(shape))


def ones(shape: ShapeLike) -> Tensor:
    """Creates a tensor of a given shape with all values being one.

    Parameters
    ----------
    ShapeLike
        Shape of the new tensor.

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    return Tensor(np.ones(shape))


def zeros_like(x: Tensor) -> Tensor:
    """Creates a tensor based on the shape of a given other tensor with all values being zero.

    Parameters
    ----------
    x : Tensor
        Tensor whose shape is used.

    Returns
    -------
    Tensor
        Tensor with all values being zero.
    """
    return Tensor(np.zeros_like(x.data))


def ones_like(x: Tensor) -> Tensor:
    """Creates a tensor based on the shape of a given other tensor with all values being one.

    Parameters
    ----------
    x : Tensor
        Tensor whose shape is used.

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    return Tensor(np.ones_like(x.data))


def randn(shape: ShapeLike) -> Tensor:
    """Creates a tensor of a given shape with random values following a normal distribution.

    Parameters
    ----------
    ShapeLike
        Shape of the new tensor.

    Returns
    -------
    Tensor
        Tensor with random values.
    """
    return Tensor(np.random.randn(*shape))


def randint(lower_bound: int, upper_bound: int, shape: ShapeLike) -> Tensor:
    """Creates a tensor of a given shape with random integer values.

    Parameters
    ----------
    lower_bound : int
        Lower bound for random values.
    upper_bound : int
        Upper bound for random values.
    ShapeLike
        Shape of the new tensor.

    Returns
    -------
    Tensor
        Tensor with random values.
    """
    return Tensor(np.random.randint(lower_bound, upper_bound, shape), dtype="int")


def shuffle(
    x1: Tensor, x2: Tensor, batch_size: int | None = None
) -> tuple[Tensor, Tensor]:
    """Shuffles two tensors equally along axis 0.

    Parameters
    ----------
    x1 : Tensor
        First tensor to be shuffled.
    x2 : Tensor
        Second tensor to be shuffled.
    batch_size : int | None, optional
        Number of samples to be returned, by default None.
        If None, all samples are returned.

    Returns
    -------
    tuple[Tensor, Tensor]
        Shuffled tensors.

    Raises
    ------
    ShapeError
        If tensors are not of equal size along a axis 0
    """
    if x1.len != x2.len:
        raise ShapeError("Tensors must have equal lengths along axis 0")

    length = x1.len
    shuffle_index = np.arange(length)
    batch_size = batch_size if batch_size else length
    np.random.shuffle(shuffle_index)
    y1 = x1[shuffle_index]
    y2 = x2[shuffle_index]
    return y1[:batch_size], y2[:batch_size]


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
    arange = np.arange(x.flatten().len)
    samples = np.random.choice(arange, size=num_samples, p=x.data.flatten())
    return Tensor(samples, dtype="int")


def empty(dtype: str = "float32") -> Tensor:
    """Return an empty tensor.

    Parameters
    ----------
    dtype: str, optional
        Datatype of the tensor data, by default float32.
    Returns
    -------
    Tensor
        Empty tensor.
    """
    return Tensor(np.empty(0, dtype=dtype))
