# weight initializations module

import numpy as np


def Random(shape, **kwargs) -> np.ndarray:
    """Returns an array with random entries between -1 and 1.

    Args:
        shape: Shape of the array.

    Returns:
        Resulting array
    """
    return np.random.randn(*shape)

def Kaiming(shape, **kwargs) -> np.ndarray:
    """Returns an array with random entries using Kaiming He initialization.

    Args:
        shape: Shape of the array.

    Kwargs:
        fan_mode: fan_mode used for Kaiming He Initialization [optional].
        activation: Activation function used [optional].

    Returns:
        Resulting array
    """
    activation = kwargs['activation']
    fan_mode = kwargs['fan_mode']
    
    gains = {
        'NoneType': 1,
        'Sigmoid': 1,
        'Tanh': 5/3,
        'Relu': 2**0.5,
        'Softmax': 1
    }

    gain = gains.get(activation.__class__.__name__, 1)
    return np.random.randn(*shape) * gain / fan_mode**0.5
