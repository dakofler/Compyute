"""Data preprocessing module"""

import pandas as pd
import numpy as np
from walnut.tensor import Tensor, ShapeLike, ShapeError


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
    shuffle_index = np.arange(len(x.data))
    np.random.shuffle(shuffle_index)
    t_shuffled = x.data[shuffle_index]
    n1 = int(len(t_shuffled) * (1 - ratio_val - ratio_test))
    n2 = int(len(t_shuffled) * (1 - ratio_test))
    train = t_shuffled[:n1]
    val = t_shuffled[n1:n2]
    test = t_shuffled[n2:]
    return Tensor(train), Tensor(val), Tensor(test)


def split_features_labels(x: Tensor, num_x_cols: int) -> tuple[Tensor, Tensor]:
    """Splits a tensor along axis 1 into two seperate tensors.

    Parameters
    ----------
    x : Tensor
        Tensor to be split.
    num_x_cols : int
        Number of feature-colums of the input tensor.

    Returns
    -------
    tuple[Tensor, Tensor]
        First tensor containing features, second tensor containing labels.
    """
    features = Tensor(x.data[:, :num_x_cols])
    labels = Tensor(x.data[:, num_x_cols:])
    return features, labels


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
    return Tensor(np.eye(num_classes)[x.data.astype("int")])


def normalize(
    x: Tensor,
    axis: ShapeLike | None = None,
    l_bound: int = -1,
    u_bound: int = 1,
) -> Tensor:
    """Normalizes a tensor using min-max feature scaling.

    Parameters
    ----------
    x : Tensor
        Tensor to be normalized.
    axis : ShapeLike | None, optional
        Axes over which normalization is applied, by default None.
        If None, the flattended tensor is normalized.
    l_bound : int, optional
        Lower bound of output values, by default -1.
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
