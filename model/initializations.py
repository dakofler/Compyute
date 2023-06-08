import numpy as np
from model import activations


def Random(shape, fan_mode: int=1, activation=activations.Identity):
    """Returns an array with random entries between -1 and 1.

    Args:
        shape: Shape of the array.
        fan_mode: Not used here [optional].
        activation: Not used here [optional].

    Returns:
        Resulting array
    """
    return np.random.randn(*shape)

def Kaiming(shape, fan_mode: int=1, activation=activations.Identity):
    """Returns an array with random entries using Kaiming initialization.

    Args:
        shape: Shape of the array.
        fan_mode: - [optional].
        activation: Activation function used [optional].

    Returns:
        Resulting array
    """
    gain = {
        activations.Identity: 1,
        activations.Sigmoid: 1,
        activations.Tanh: 5/3,
        activations.ReLu: 2**0.5,
        activations.Softmax: 1
    }
    return np.random.randn(*shape) * gain[activation] / fan_mode**0.5