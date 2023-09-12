"""Tensor utils module"""

import pandas as pd
import numpy as np
from walnut.cuda import get_cpt_pkg
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
    cpt_pkg = get_cpt_pkg(x.device)
    return Tensor(
        cpt_pkg.expand_dims(x.data, axis=axis), dtype=x.dtype, device=x.device
    )


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
    cpt_pkg = get_cpt_pkg(device)
    x = cpt_pkg.arange(start, stop, step)
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
    cpt_pkg = get_cpt_pkg(device)
    return Tensor(cpt_pkg.linspace(start, stop, num), device=device)


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
    cpt_pkg = get_cpt_pkg(device)
    return Tensor(cpt_pkg.zeros(shape), device=device)


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
    cpt_pkg = get_cpt_pkg(device)
    return Tensor(cpt_pkg.ones(shape), device=device)


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
    cpt_pkg = get_cpt_pkg(device)
    return Tensor(cpt_pkg.random.normal(mean, std, shape), device=device)


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
    cpt_pkg = get_cpt_pkg(device)
    return Tensor(cpt_pkg.random.uniform(low, high, shape), device=device)


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
    cpt_pkg = get_cpt_pkg(device)
    return Tensor(cpt_pkg.random.randint(low, high, shape), dtype="int", device=device)


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
    cpt_pkg = get_cpt_pkg(device)
    return Tensor(cpt_pkg.random.permutation(n), dtype="int", device=device)


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
    cpt_pkg = get_cpt_pkg(p.device)
    return Tensor(
        cpt_pkg.random.choice(p.len, size=num_samples, p=p.data),
        dtype="int",
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
    cpt_pkg = get_cpt_pkg(device)
    return Tensor(
        cpt_pkg.random.choice(x.data, shape, p=p.data), dtype=x.dtype, device=device
    )


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
    cpt_pkg = get_cpt_pkg(device)
    emp = cpt_pkg.empty(0, dtype=dtype)
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

    cpt_pkg = get_cpt_pkg(device)
    return Tensor(cpt_pkg.maximum(_a, _b), device=device)


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
    cpt_pkg = get_cpt_pkg(device)
    return Tensor(
        cpt_pkg.concatenate([t.data for t in tensors], axis=axis),
        device=device,
        dtype=tensors[0].dtype,
    )


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
    cpt_pkg = get_cpt_pkg(device)
    return Tensor(cpt_pkg.eye(n), device=device)


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
    cpt_pkg = get_cpt_pkg(x.device)
    split_data = cpt_pkg.split(x.data, splits, axis=axis)
    return [Tensor(s, dtype=x.dtype, device=x.device) for s in split_data]
