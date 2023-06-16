# utility functions module

import pandas as pd
import numpy as np
from typing import Union


def split_train_test_val_data(array: np.ndarray, ratio_val: float=0.1, ratio_test: float=0.1) -> (np.ndarray|np.ndarray|np.ndarray):
    """Splits an array along axis 0 into three seperate arrays using a given ratio.

    Args:
        array: Array to be split.
        ratio_val: Size ratio of the validation split [optional].
        ratio_test: Size ratio of the test split [optional].
    
    Returns:
        train_array: first array
        test_array: second array
    """
    shuffle_index = np.arange(len(array))
    np.random.shuffle(shuffle_index)
    array_shuffled = array[shuffle_index]

    n1 = int(len(array_shuffled) * (1 - ratio_val - ratio_test))
    n2 = int(len(array_shuffled) * (1 - ratio_val))

    train_array = array_shuffled[:n1]
    val_array = array_shuffled[n1:n2]
    test_array = array_shuffled[n2:]
    
    return train_array, val_array, test_array

def expand_dims(array: np.ndarray, dims: int) -> np.ndarray:
    """Extends the dimension of an array.

    Args:
        array: Array to be extended.
        dims: Number of dimensions of the desired output array.
    
    Returns:
        Array with extended dimensions.
    """
    while array.ndim < dims:
        array = np.expand_dims(array, -1)
    return array

def split_X_Y(array: np.ndarray, num_x_cols: int) -> (np.ndarray|np.ndarray):
    """Splits an array along axis 1 into two seperate arrays.

    Args:
        array: Array to be split.
        num_x_cols: Number of feature-colums of the input array.
    
    Returns:
        X: feature-array
        Y: class-array
    """
    X = array[:, :num_x_cols]
    Y = array[:, num_x_cols:]
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

def one_hot_encode(array: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot-encodes an array of shape (n,) into an array of shape (n, num_classes).

    Args:
        array: Array containing categorical columns.
        num_classes: number of classes to be considered when encoding. Defines axis 1 of the output array.
    
    Returns:
        array: One-hot-encoded array.
    """
    return np.eye(num_classes)[array]

def normalize(x: np.ndarray, axis: Union[int, tuple[int, int]]=0, a: int=-1, b: int=1) -> np.ndarray:
    """Normalizes an array along a certain axis using min-max feature scaling.

    Args:
        x: Array to be normalized.
        axis: One or multiple axes along which values are to be normalized [optional].
        a: Lower bound of output values [optional].
        b: Upper bound of output values [optional].
    
    Returns:
        x_normalized: Normalized array.
    """
    x_min = x.min(axis=axis)
    x_max = x.max(axis=axis)
    return a + (x - x_min) * (b - a) / (x_max - x_min)
