import numpy as np


def ReLu(v: np.ndarray, derivative: bool=False):
    """Applies the Rectified Linear Unit function to an array.

    Args:
        v: Array the function is to be applied to.
        derivatie: Whether the derivative of the function shall be applied instead [optional].

    Returns:
        Resulting array
    """
    if not derivative:
        return np.maximum(0, v)
    else:
        return (v > 0).astype(int)

def LeakyReLu(v: np.ndarray, derivative: bool=False):
    """Applies the Leaky Rectified Linear Unit function to an array.

    Args:
        v: Array the function is to be applied to.
        derivatie: Whether the derivative of the function shall be applied instead [optional].

    Returns:
        Resulting array
    """
    if not derivative:
        return np.maximum(0.01 * v, v)
    else:
        d = (v > 0).astype(int)
        d[d > 0] = 0.01
        return d

def Identity(v: np.ndarray, derivative: bool=False):
    """Applies the Identity function to an array.

    Args:
        v: Array the function is to be applied to.
        derivatie: Whether the derivative of the function shall be applied instead [optional].

    Returns:
        Resulting array
    """
    if not derivative:
        return v
    else:
        return np.ones(v.shape)
    
def Sigmoid(v: np.ndarray, derivative: bool=False):
    """Applies the Sigmoid function to an array.

    Args:
        v: Array the function is to be applied to.
        derivatie: Whether the derivative of the function shall be applied instead [optional].

    Returns:
        Resulting array
    """
    if not derivative:
        v = np.clip(v, -100, 100)
        return 1.0 / (1.0 + np.exp(-v))
    else:
        return Sigmoid(v) * (1.0 - Sigmoid(v))

def Tanh(v: np.ndarray, derivative: bool=False):
    """Applies the Tangens Hyperbolicus function to an array.

    Args:
        v: Array the function is to be applied to.
        derivatie: Whether the derivative of the function shall be applied instead [optional].

    Returns:
        Resulting array
    """
    if not derivative:
        return np.tanh(v)
    else:
        return 1.0 - (Tanh(v) * Tanh(v))

def Softmax(v: np.ndarray, derivative: bool=False):
    """Applies the Softmax function to an array.

    Args:
        v: Array the function is to be applied to.
        derivatie: Whether the derivative of the function shall be applied instead [optional].

    Returns:
        Resulting array
    """
    if not derivative:
        e = np.exp(v - np.max(v))
        return e / np.sum(e, axis=0)
    else:
        softmax = np.reshape(Softmax(v), (1, -1))
        return (softmax * np.identity(softmax.size) - softmax.transpose() @ softmax)