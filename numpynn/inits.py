"""tensor initializations module"""

import numpy as np


def random(shape):
    """Creates a tensor of a given shape following a normal distribution.

    Args:
        shape: Shape of the tensor.

    Returns:
        Tensor with random values.
    """
    return np.random.randn(*shape).astype('float32')

def kaiming(shape, **kwargs):
    """Creates a tensor of a given shape with values according to the Kaiming He initialization.

    Args:
        shape: Shape of the tensor.

    Kwargs:
        fan_mode: Integer value used for Kaiming initialization [optional].
        activation: Activation function used [optional].

    Returns:
        Tensor with random values.
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
    return (np.random.randn(*shape) * gain / fan_mode**0.5).astype('float32')

def zeros(shape):
    """Creates a tensor of a given shape with all values being zero.

    Args:
        shape: Shape of the tensor.

    Returns:
        Tensor with all values being zero.
    """
    return np.zeros(shape, dtype='float32')

def ones(shape):
    """Creates a tensor of a given shape with all values being one.

    Args:
        shape: Shape of the tensor.

    Returns:
        Tensor with all values being one.
    """
    return np.ones(shape, dtype='float32')

def zeros_like(tensor):
    """Creates a tensor of shape of a given other tensor with all values being zero.

    Args:
        tensor: Tensor whose shape is to be used.

    Returns:
        Tensor with all values being zero.
    """
    return np.zeros_like(tensor, dtype='float32')

def ones_like(tensor):
    """Creates a tensor of shape of a given other tensor with all values being one.

    Args:
        tensor: Tensor whose shape is to be used.

    Returns:
        Tensor with all values being one.
    """
    return np.ones_like(tensor, dtype='float32')
