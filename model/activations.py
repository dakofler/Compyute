import numpy as np

def ReLu(v):
    return np.maximum(0.0, v)

def Identity(v):
    return v
    
def Sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))

def Tanh(v):
    return np.tanh(v)

def Softmax(v):
    return np.exp(v) / np.sum(np.exp(v))