from calendar import day_abbr
import pandas as pd
import numpy as np

def split_train_val_data(data, ratio=0.3):
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