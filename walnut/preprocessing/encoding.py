"""data encoding module"""

import pandas as pd
import numpy as np
import cupy as cp

from walnut.tensor import Tensor
import walnut.tensor_utils as tu


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


def pd_categorical_to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
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
    df[columns] = df[columns].astype("category")
    df[columns] = df[columns].apply(lambda x: x.cat.codes)
    return df


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
    if x.dtype not in ("int", "int32", "int64"):
        raise ValueError(f'Invalid datatype {x.dtype}. Must be "int".')
    return Tensor((tu.eye(num_classes, x.device)[x]).data, dtype="int", device=x.device)
