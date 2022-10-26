import numpy as np

def ReLu(v, derivative = False):
    if not derivative:
        return np.maximum(0.0, v)
    else:
        a = v.copy()
        a[a <= 0.0] = 0.0
        a[a > 0.0] = 1.0
        return a

def Identity(v, derivative = False):
    if not derivative:
        return v
    else:
        return np.ones(v.shape)
    
def Sigmoid(v, derivative = False):
    if not derivative:
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
        return np.exp(v) / np.sum(np.exp(v))
    else:
        # https://e2eml.school/softmax.html
        softmax = np.reshape(Softmax(v), (1, -1))
        return (softmax * np.identity(softmax.size) - softmax.transpose() @ softmax)
