"""basic preprocessing module"""

from .._tensor_functions import identity
from .._types import _AxisLike
from ..random import shuffle
from ..tensors import Tensor

__all__ = ["split_train_val_test", "normalize", "standardize", "one_hot_encode"]


def split_train_val_test(
    x: Tensor, ratio_val: float = 0.1, ratio_test: float = 0.1
) -> tuple[Tensor, Tensor, Tensor]:
    """Splits a tensor along axis 0 into three seperate tensor using a given ratio.

    Parameters
    ----------
    x : Tensor
        Tensor to be split.
    ratio_val : float, optional
        Size ratio of the validation split, by default 0.1.
    ratio_test : float, optional
        Size ratio of the test split, by default 0.1.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Train, validation and test tensors.
    """
    x_shuffled = shuffle(x)[0]
    n1 = int(len(x_shuffled) * (1 - ratio_val - ratio_test))
    n2 = int(len(x_shuffled) * (1 - ratio_test))
    train = x_shuffled[:n1]
    val = x_shuffled[n1:n2]
    test = x_shuffled[n2:]
    return train, val, test


def normalize(
    x: Tensor,
    axis: _AxisLike | None = None,
    l_bound: int = 0,
    u_bound: int = 1,
) -> Tensor:
    """Normalizes a tensor using min-max feature scaling.

    Parameters
    ----------
    x : Tensor
        Tensor to be normalized.
    axis : AxisLike | None, optional
        Axes over which normalization is applied, by default None.
        If None, the flattended tensor is normalized.
    l_bound : int, optional
        Lower bound of output values, by default 0.
    u_bound : int, optional
        Upper bound of output values, by default 1.

    Returns
    -------
    Tensor
        Normalized tensor.
    """

    x_min = x.min(axis=axis)
    x_max = x.max(axis=axis)
    return (x - x_min) * (u_bound - l_bound) / (x_max - x_min) + l_bound


def standardize(x: Tensor, axis: _AxisLike | None = None) -> Tensor:
    """Standardizes a tensor to mean 0 and variance 1.

    Parameters
    ----------
    x : Tensor
        Tensor to be standardized.
    axis : AxisLike | None, optional
        Axes over which standardization is applied, by default None.
        If None, the flattended tensor is standardized.

    Returns
    -------
    Tensor
        Standardized tensor with mean 0 and variance 1.
    """

    return (x - x.mean(axis=axis)) / x.var(axis=axis)


def one_hot_encode(x: Tensor, num_classes: int) -> Tensor:
    """One-hot-encodes a tensor, by adding an additional encoding dimension.

    Parameters
    ----------
    x : Tensor
        Tensor containing categorical columns of type `int`.
    num_classes : int
        Number of classes to be considered when encoding.
        Defines axis -1 of the output tensor.

    Returns
    -------
    Tensor
        One-hot-encoded tensor.

    Raises
    -------
    ValueError
        If the tensor dtype is not int.
    """
    if x.dtype not in {"int", "int8", "int16", "int32", "int64"}:
        raise ValueError(f'Invalid datatype {x.dtype}. Must be "int".')

    return identity(num_classes, "int32", x.device)[x]
