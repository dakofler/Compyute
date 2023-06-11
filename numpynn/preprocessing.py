# utility functions module

import pandas as pd
import numpy as np
from typing import Union


def split_train_test_data(array: np.ndarray, ratio: float=0.3) -> (np.ndarray|np.ndarray):
    """Splits an array along axis 0 into two seperate arrays using a given ratio, which defines the latter arrays' length.

    Args:
        array: Array to be split.
        ratio: Ratio between input array size and the second arrays' size [optional].
    
    Returns:
        train_array: first array
        test_array: second array
    """
    shuffle_index = np.arange(len(array))
    np.random.shuffle(shuffle_index)
    array_shuffled = array[shuffle_index]
    i = int(len(array_shuffled) * (1 - ratio))
    train_array = array_shuffled[:i]
    test_array = array_shuffled[i:]
    return train_array, test_array

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

def categorical_to_numeric(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Transforms categorical columns of a dataframe into numerical columns.

    Args:
        dataframe: Dataframe containing categorical columns.
    
    Returns:
        numerical_dataframe: Dataframe with transformed categorical columns.
    """
    return pd.get_dummies(dataframe)

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
