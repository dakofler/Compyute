"""padding functions module"""

import math
import numpy as np
from numpynn.tensor import Tensor


def valid(tensor: Tensor, **kwargs) -> Tensor:
    """Applies valid padding using zero-values to a tensor.

    Args:
        tensor: Tensor padding is to be applied to.

    Returns:
        Padded tensor.
    """
    return tensor

def same(tensor: Tensor, **kwargs) -> Tensor:
    """Applies same padding using zero-values to a tensor.

    Args:
        tensor: Tensor padding is to be applied to.

    Kwargs:
        kernel_shape: Kernel shape the padding width is adapted to.

    Returns:
        Padded tensor.
    """
    kernel_shape = kwargs['kernel_shape']
    width = math.floor(kernel_shape[0] / 2)
    # pad along axis 2 & 3
    return Tensor(np.pad(tensor.data, ((0, 0), (0, 0), (width, width), (width, width))))
