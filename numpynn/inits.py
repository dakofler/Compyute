"""tensor initializations module"""

import numpy as np
from numpynn.tensor import Tensor


def random(shape) -> Tensor:
    """Creates a tensor of a given shape following a normal distribution.

    Args:
        shape: Shape of the tensor.

    Returns:
        Tensor with random values.
    """
    return Tensor(np.random.randn(*shape))

def kaiming(shape, **kwargs) -> Tensor:
    """Creates a tensor of a given shape with values according to the Kaiming He initialization.

    Args:
        shape: Shape of the tensor.

    Kwargs:
        fan_mode: Integer value used for Kaiming initialization [optional].
        activation: Activation function used [optional].

    Returns:
        Tensor with random values.
    """
    act_fn = kwargs['act_fn']
    fan_mode = kwargs['fan_mode']
    gains = {
        'NoneType': 1,
        'Sigmoid': 1,
        'Tanh': 5/3,
        'Relu': 2**0.5,
        'Softmax': 1
    }
    gain = gains.get(act_fn.__class__.__name__, 1)
    return Tensor(np.random.randn(*shape) * gain / fan_mode**0.5)
