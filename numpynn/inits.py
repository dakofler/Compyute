import numpy as np


def Random(shape, fan_mode: int=1, activation=None) -> np.ndarray:
    """Returns an array with random entries between -1 and 1.

    Args:
        shape: Shape of the array.
        fan_mode: Not used here [optional].
        activation: Not used here [optional].

    Returns:
        Resulting array
    """
    return np.random.randn(*shape)

def Kaiming(shape, fan_mode: int=1, activation=None) -> np.ndarray:
    """Returns an array with random entries using Kaiming initialization.

    Args:
        shape: Shape of the array.
        fan_mode: - [optional].
        activation: Activation function used [optional].

    Returns:
        Resulting array
    """
    gains = {
        'NoneType': 1,
        'Sigmoid': 1,
        'Tanh': 5/3,
        'Relu': 2**0.5,
        'Softmax': 1
    }

    gain = gains.get(activation.__class__.__name__, 1)
    return np.random.randn(*shape) * gain / fan_mode**0.5