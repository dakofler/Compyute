import numpy as np


def Random(shape, fan_mode: int=1):
    """Returns an array with random entries between -1 and 1.

    Args:
        shape: Shape of the array.
        fan_mode: Not used here [optional].

    Returns:
        Resulting array
    """
    return np.random.uniform(-1.0, 1.0, shape)

def Kaiming(shape, fan_mode: int=1):
    """Returns an array with random entries using Kaiming initialization.

    Args:
        shape: Shape of the array.
        fan_mode: - [optional].

    Returns:
        Resulting array
    """
    return np.random.uniform(-1.0, 1.0, shape) / fan_mode**0.5