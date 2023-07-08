"""data encoding module"""

import pandas as __pd
import numpy as __np
from walnut.tensor import (
    Tensor as __Tensor,
    ShapeError as __ShapeError,
    NumpyArray as __NumpyArray,
)

__all__ = ["pd_one_hot_encode", "one_hot_encode"]


def pd_one_hot_encode(df: __pd.DataFrame, columns: list[str]) -> __pd.DataFrame:
    """One-hot-encodes categorical columns of a dataframe into numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing categorical columns.
    columns : list[str]
        Names of columns to be encoded.

    Returns
    -------
    pd.DataFrame
        Dataframe containing transformed categorical columns.
    """
    return __pd.get_dummies(df, columns=columns)


def one_hot_encode(x: __Tensor, num_classes: int) -> __Tensor:
    """One-hot-encodes a one-dimensional tensor.

    Parameters
    ----------
    x : Tensor
        Tensor containing categorical columns of type `int`.
    num_classes : int
        Number of classes to be considered when encoding.
        Defines axis 1 of the output tensor.

    Returns
    -------
    Tensor
        One-hot-encoded tensor of shape (n, num_classes).
    """
    if x.ndim != 1:
        raise __ShapeError("Tensor must be of dim 1.")
    return __Tensor(__np.eye(num_classes)[x.data.astype("int")])


def list_one_hot_encode(x: list[int], num_classes: int) -> __NumpyArray:
    """One-hot-encodes a list of ints.

    Parameters
    ----------
    x : list[int]
        List containing categorical columns of type `int`.
    num_classes : int
        Number of classes to be considered when encoding.
        Defines axis 1 of the output tensor.

    Returns
    -------
    NumpyArray
        One-hot-encoded array of shape (n, num_classes).
    """
    return __np.eye(num_classes)[x]
