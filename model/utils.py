import pandas as pd

def splitData(data, X, Y):
    pass

def splitTrainValData(data, ratio=0.3):
    val_data = data.sample(n=int(len(data.index) * ratio))
    train_data = data.drop(val_data.index)
    return train_data, val_data

def splitXY(data, X_cols, Y_cols):
    X = data[[X_cols]]
    Y = data[[Y_cols]]
    return X, Y