import pandas as pd
import numpy as np


def df_split_train_val_data(data, ratio=0.3):
    val_data = data.sample(n=int(len(data.index) * ratio))
    train_data = data.drop(val_data.index)
    return train_data, val_data

def split_X_Y(data, num_x_cols):
    d = np.split(data, (num_x_cols, num_x_cols * 2), axis=1)
    return d[0], d[1]

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