"""padding functions module"""

import math
import numpy as np


def valid(tensor: np.ndarray, **kwargs) -> np.ndarray:
    """Applies valid padding using zero-values to a tensor.

    Args:
        tensor: Tensor padding is to be applied to.

    Returns:
        Padded tensor.
    """
    return tensor.copy()

def same(tensor: np.ndarray, **kwargs) -> np.ndarray:
    """Applies same padding using zero-values to a tensor.

    Args:
        tensor: Tensor padding is to be applied to.

    Kwargs:
        kernel_size: Kernel size the padding width is adapted to.

    Returns:
        Padded tensor.
    """
    kernel_dim = kwargs['kernel_dim']
    width = math.floor(kernel_dim[0] / 2)
    # pad along axis 2 & 3
    return np.pad(tensor, ((0, 0), (0, 0), (width, width), (width, width)))
