"""data encoding module"""

import pandas as pd
import numpy as np
from walnut.tensor import Tensor, ShapeError

__all__ = ["pd_one_hot_encode", "one_hot_encode"]


def pd_one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
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
    return pd.get_dummies(df, columns=columns)


def one_hot_encode(x: Tensor, num_classes: int) -> Tensor:
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
        raise ShapeError("Tensor must be of dim 1.")
    return Tensor(np.eye(num_classes)[x.data], dtype="int")
