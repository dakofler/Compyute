import pandas as pd
import numpy as np
from numpy.fft  import fft2, ifft2
from scipy import signal
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

def shuffle(x: np.ndarray, y: np.ndarray) -> (np.ndarray|np.ndarray):
    """Shuffles two arrays along a axis 0 equally.

    Args:
        x: First array to be shuffled.
        x: First array to be shuffled.
    
    Returns:
        x_shuffled: First shuffled array.
        y_shuffled: Second shuffled array.
    """
    shuffle_index = np.arange(len(x))
    np.random.shuffle(shuffle_index)
    x_shuffled = x[shuffle_index]
    y_shuffled = y[shuffle_index]
    return x_shuffled, y_shuffled

def convolve_loop(array: np.ndarray, filter: np.ndarray, stride: int=1) -> np.ndarray:
    """Performs a convolution operation of a 2D array and a 2D filter using loops.

    Args:
        array: Array to be convolved.
        filter: Filter-array used for the convolution.
        stride: Stride value used in the operation [optional].    

    Returns:
        array_convolved: Resulting array of the conolution operation.

    Raises:
        ShapeError: If array and/or filter are not of dim 2.
    """
    if array.ndim != 2 or filter.ndim != 2: raise Exception('ShapeError: Array and filter must be of dim 2.')
    o_y = int((array.shape[0] - filter.shape[0]) / stride) + 1
    f_y = filter.shape[0]
    f_x = filter.shape[1]
    o = np.zeros((o_y, o_y))
    filter = np.fliplr(filter)

    y_count = 0
    for y in range(0, o_y * stride, stride):
        x_count = 0
        for x in range(0, o_y * stride, stride):
            chunk = array[y : y + f_y, x : x + f_x]
            o[y_count, x_count] = np.sum(chunk * filter)
            x_count += 1
        y_count += 1
    return o

def convolve_scipy(array: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """Performs a convolution operation of a 2D array and a 2D filter using SciPy's convolve2d method.

    Args:
        array: Array to be convolved.
        filter: Filter-array used for the convolution.
    
    Returns:
        array_convolved: Resulting array of the conolution operation.

    Raises:
        ShapeError: If array and/or filter are not of dim 2.
    """
    if array.ndim != 2 or filter.ndim != 2: raise Exception('ShapeError: Array and filter must be of dim 2.')
    return signal.convolve2d(array, filter, mode='valid')

def convolve_fft(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Performs a convolution operation of a 2D array and a 2D filter using FFT.

    Args:
        array: Array to be convolved.
        filter: Filter-array used for the convolution.
    
    Returns:
        array_convolved: Resulting array of the conolution operation.

    Raises:
        ShapeError: If array and/or filter are not of dim 2.
    """
    if array.ndim != 2 or filter.ndim != 2: raise Exception('ShapeError: Array and filter must be of dim 2.')
    kernel_fft = fft2(kernel, s=array.shape)
    array_fft = fft2(array)
    return np.real(ifft2(array_fft * kernel_fft))