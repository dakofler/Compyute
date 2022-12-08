import pandas as pd
import numpy as np
from scipy import signal


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

def convolve(image: np.ndarray, f: np.ndarray, s: int = 1):
    # o_y = int((image.shape[0] - f.shape[0]) / s) + 1
    # f_y = f.shape[0]
    # f_x = f.shape[1]
    # o = np.zeros((o_y, o_y))
    # f = np.fliplr(f)

    # y_count = 0
    # for y in range(0, o_y * s, s):
    #     x_count = 0
    #     for x in range(0, o_y * s, s):
    #         array = image[y : y + f_y, x : x + f_x]
    #         o[y_count, x_count] = np.sum(array * f)
    #         x_count += 1
    #     y_count += 1
    # return o

    return signal.convolve2d(image, f, mode='valid')