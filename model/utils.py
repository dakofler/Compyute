import pandas as pd
import numpy as np


def split_train_test_data(array, ratio=0.3):
    shuffle_index = np.arange(len(array))
    np.random.shuffle(shuffle_index)
    array_shuffled = array[shuffle_index]
    i = int(len(array_shuffled) * (1 - ratio))
    train_array = array_shuffled[:i]
    test_array = array_shuffled[i:]
    return train_array, test_array

def expand_dims(array: np.ndarray, dims):
    while array.ndim < dims:
        array = np.expand_dims(array, -1)
    return array

def split_X_Y(data, num_x_cols):
    X = data[:, :num_x_cols]
    Y = data[:, num_x_cols:]
    return X, Y

def categorical_to_numeric(data: pd.DataFrame):
    return pd.get_dummies(data)

def normalize(array: np.ndarray):
    return array / array.max(axis=0)

def shuffle(x: np.ndarray, y: np.ndarray):
    shuffle_index = np.arange(len(x))
    np.random.shuffle(shuffle_index)
    x_shuffled = x[shuffle_index]
    y_shuffled = y[shuffle_index]
    return x_shuffled, y_shuffled