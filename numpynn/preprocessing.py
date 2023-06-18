"""utility functions module"""

from typing import Union
import pandas as pd
import numpy as np


def split_train_val_test(tensor: np.ndarray, ratio_val: float=0.1,
                        ratio_test: float=0.1) -> (np.ndarray|np.ndarray|np.ndarray):
    """Splits a tensor along axis 0 into three seperate tensor using a given ratio.

    Args:
        tensor: Tensor to be split.
        ratio_val: Size ratio of the validation split [optional].
        ratio_test: Size ratio of the test split [optional].
    
    Returns:
        train: first tensor
        val: second tensor
        test: third tensor
    """
    shuffle_index = np.arange(len(tensor))
    np.random.shuffle(shuffle_index)
    t_shuffled = tensor[shuffle_index]
    n1 = int(len(t_shuffled) * (1 - ratio_val - ratio_test))
    n2 = int(len(t_shuffled) * (1 - ratio_test))
    train = t_shuffled[:n1]
    val = t_shuffled[n1:n2]
    test = t_shuffled[n2:]
    return train, val, test

def split_features_labels(tensor: np.ndarray, num_x_cols: int) -> (np.ndarray|np.ndarray):
    """Splits a tensor along axis 1 into two seperate tensors.

    Args:
        tensor: Tensor to be split.
        num_x_cols: Number of feature-colums of the input tensor.
    
    Returns:
        X: feature-tensor
        Y: label-tensor
    """
    X = tensor[:, :num_x_cols]
    Y = tensor[:, num_x_cols:]
    return X, Y

def categorical_to_numeric(dataframe: pd.DataFrame, columns: list[str]=None) -> pd.DataFrame:
    """Transforms categorical columns of a dataframe into numerical columns.

    Args:
        dataframe: Dataframe containing categorical columns.
        columns: Columns to be encoded.
    
    Returns:
        numerical_dataframe: Dataframe with transformed categorical columns.
    """
    return pd.get_dummies(dataframe, columns=columns)

def one_hot_encode(tensor: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot-encodes a tensor of shape (n,) into a tensor of shape (n, num_classes).

    Args:
        tensor: Tensor containing categorical columns.
        num_classes: number of classes to be considered when encoding.
            Defines axis 1 of the output tensor.
    
    Returns:
        One-hot-encoded tensor.
    """
    return (np.eye(num_classes)[tensor]).astype('float32')

def normalize(tensor: np.ndarray, axis: Union[int, tuple[int, int]]=0,
              l_bound: int=-1, u_bound: int=1) -> np.ndarray:
    """Normalizes a tensor along a certain axis using min-max feature scaling.

    Args:
        tensor: Tensor to be normalized.
        axis: One or multiple axes along which values are to be normalized [optional].
        l_bound: Lower bound of output values [optional].
        u_bound: Upper bound of output values [optional].
    
    Returns:
        x_normalized: Normalized tensor.
    """
    t_min = tensor.min(axis=axis)
    t_max = tensor.max(axis=axis)
    return l_bound + (tensor - t_min) * (u_bound - l_bound) / (t_max - t_min)
