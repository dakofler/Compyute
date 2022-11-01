import numpy as np


def error_dynamic(eta, loss_hist, dynamic_range=10):
    if len(loss_hist) < dynamic_range + 1:
        return eta
    dynamic = np.sum((np.array(loss_hist[-dynamic_range:]) - np.array(loss_hist[(-dynamic_range - 1):-1])) / np.absolute(np.array(loss_hist[(-dynamic_range - 1):-1]))) / dynamic_range
    return eta / 2.0 if dynamic > 0 else eta

def none(eta, loss_hist, dynamic_range=10):
    return eta