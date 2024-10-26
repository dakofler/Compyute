"""Basic data preprocessing utilities."""

from typing import Optional

from ..random.random import shuffle
from ..tensor_ops.creation_ops import identity
from ..tensors import DimLike, Tensor
from ..typing import DType, int64, is_integer

__all__ = ["split_train_val_test", "normalize", "standardize", "one_hot_encode"]


def split_train_val_test(
    x: Tensor, ratio_val: float = 0.1, ratio_test: float = 0.1
) -> tuple[Tensor, Tensor, Tensor]:
    """Splits a tensor along the frist dimension into three
    seperate tensors using a given ratio.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    ratio_val : float, optional
        Size ratio of the validation split. Defaults to ``0.1``.
    ratio_test : float, optional
        Size ratio of the test split. Defaults to ``0.1``.

    Returns
    -------
    Tensor
        Train split.
    Tensor
        Validation split.
    Tensor
        Test split.
    """
    x_shuffled = shuffle(x)[0]
    n1 = int(len(x_shuffled) * (1.0 - ratio_val - ratio_test))
    n2 = int(len(x_shuffled) * (1.0 - ratio_test))
    train = x_shuffled[:n1]
    val = x_shuffled[n1:n2]
    test = x_shuffled[n2:]
    return train, val, test


def normalize(
    x: Tensor,
    dim: Optional[DimLike] = None,
    low: float = 0.0,
    high: float = 1.0,
) -> Tensor:
    """Normalizes a tensor using min-max feature scaling.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : DimLike, optional
        Dimension on which to perform the operation. Defaults to ``None``.
        If ``None`` it is performed on the flattened tensor.
    low : float, optional
        Lower bound of output values. Defaults to ``0``.
    high : float, optional
        Upper bound of output values. Defaults to ``1``.

    Returns
    -------
    Tensor
        Normalized tensor.
    """
    x_min, x_max = x.min(dim), x.max(dim)
    return (x - x_min) * (high - low) / (x_max - x_min) + low


def standardize(
    x: Tensor,
    dim: Optional[DimLike] = None,
) -> Tensor:
    """Standardizes a tensor to mean 0 and variance 1.

    Parameters
    ----------
    x : Tensor
        Tensor to be standardized.
    dim : DimLike, optional
        Dimension on which to perform the operation. Defaults to ``None``.
        If ``None`` it is performed on the flattened tensor.

    Returns
    -------
    Tensor
        Standardized tensor with mean ``0`` and variance ``1``.
    """
    return (x - x.mean(dim)) / x.std(dim)


def one_hot_encode(
    x: Tensor, num_classes: int, dtype: Optional[DType] = int64
) -> Tensor:
    """One-hot-encodes a tensor, by adding an additional encoding dimension.

    Parameters
    ----------
    x : Tensor
        Tensor containing categorical columns of type ``int``.
    num_classes : int
        Number of classes to be considered when encoding.
        Defines dim ``-1`` of the output tensor.
    dtype : DtypeLike, optional
        Datatype of the tensor data. Defaults to ``None``.

    Returns
    -------
    Tensor
        One-hot-encoded tensor.

    Raises
    -------
    ValueError
        If the tensor dtype is not ``int``.
    """
    if not is_integer(x.dtype):
        raise ValueError(f"Input must be an integer, got '{x.dtype}'.")

    return identity(num_classes, device=x.device, dtype=dtype)[x]
