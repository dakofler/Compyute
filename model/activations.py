import numpy as np

def ReLu(v, derivative = False):
    if not derivative: return np.maximum(0.0, v)
    else: return 0

def Identity(v, derivative = False):
    if not derivative: return v
    else: return 1
    
def Sigmoid(v, derivative = False):
    if not derivative: return 1.0 / (1.0 + np.exp(-v))
    else: return 0

def Tanh(v, derivative = False):
    if not derivative: return np.tanh(v)
    else: return 0

def Softmax(v, derivative = False):
    if not derivative: return np.exp(v) / np.sum(np.exp(v))
    else: return 0