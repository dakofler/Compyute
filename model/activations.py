import numpy as np


def ReLu(v, derivative = False):
    if not derivative:
        return np.maximum(0, v)
    else:
        return (v > 0).astype(int)


def LeakyReLu(v, derivative = False):
    if not derivative:
        return np.maximum(0.01 * v, v)
    else:
        d = (v > 0).astype(int)
        d[d > 0] = 0.01
        return d


def Identity(v, derivative = False):
    if not derivative:
        return v
    else:
        return np.ones(v.shape)
    
def Sigmoid(v: np.ndarray, derivative = False):
    if not derivative:
        v = np.clip(v, -100, 100) # set min and max, because sigmoid can overflow
        return 1.0 / (1.0 + np.exp(-v))
    else:
        return Sigmoid(v) * (1.0 - Sigmoid(v))

def Tanh(v, derivative = False):
    if not derivative:
        return np.tanh(v)
    else:
        return 1.0 - (Tanh(v) * Tanh(v))

def Softmax(v, derivative = False):
    if not derivative:
        # return np.exp(v) / np.sum(np.exp(v))
        # sometimes got an overflow, found this solution by Shusei Eshima https://shusei-e.github.io/deep%20learning/softmax_without_overflow/
        e = np.exp(v - np.max(v))
        return e / np.sum(e, axis=0)

    else:
        # https://e2eml.school/softmax.html
        softmax = np.reshape(Softmax(v), (1, -1))
        return (softmax * np.identity(softmax.size) - softmax.transpose() @ softmax)
