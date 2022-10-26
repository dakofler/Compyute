import numpy as np

def error_dynamic(learning_rate, loss_hist, dynamic_range=10):
    if len(loss_hist) < dynamic_range + 1:
        return learning_rate
    dynamic = np.sum((np.array(loss_hist[-dynamic_range:]) - np.array(loss_hist[(-dynamic_range - 1):-1])) / np.absolute(np.array(loss_hist[(-dynamic_range - 1):-1]))) / dynamic_range
    return learning_rate / 2.0 if dynamic > 0 else learning_rate

def none(learning_rate, loss_hist, dynamic_range=10):
    return learning_rate