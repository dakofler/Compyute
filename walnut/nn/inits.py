"""tensor initializations module"""

from walnut import tensor
from walnut.tensor import Tensor


def random(shape: tuple[int]) -> Tensor:
    """Creates a tensor of a given shape following a normal distribution.

    ### Parameters
        shape: `tuple[int]`
            Shape of the new tensor.

    ### Returns
        y: `Tensor`
            Tensor with random values.
    """
    return tensor.randn(shape)

def kaiming(shape: tuple[int], **kwargs) -> Tensor:
    """Creates a tensor of a given shape with values using Kaiming He initialization.

    ### Parameters
        shape: `tuple[int]`
            Shape of the new tensor.
        **kwargs:
            fan_mode: `int`
                Integer value used for Kaiming initialization.
            activation: `Activation`
                Activation function used.

    ### Returns
        y: `Tensor`
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
    return tensor.randn(shape) * gain / fan_mode**0.5
