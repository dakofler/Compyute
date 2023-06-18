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
        kernel_shape: Kernel shape the padding width is adapted to.

    Returns:
        Padded tensor.
    """
    kernel_shape = kwargs['kernel_shape']
    width = math.floor(kernel_shape[0] / 2)
    # pad along axis 2 & 3
    return np.pad(tensor, ((0, 0), (0, 0), (width, width), (width, width)))
