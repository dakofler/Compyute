# padding functions module

import numpy as np
import math


def Valid(array: np.ndarray, **kwargs) -> np.ndarray:
    """Applies Valid padding using zero-values to an input array.

    Args:
        array: Array padding is to be applied to.

    Returns:
        Padded array.
    """
    return array.copy()

def Same(array: np.ndarray, **kwargs) -> np.ndarray:
    """Applies Same padding using zero-values to an input array.

    Args:
        array: Array padding is to be applied to.

    Kwargs:
        kernel_size: Kernel size the padding width is adapted to.

    Returns:
        Padded array.
    """
    kernel_size = kwargs['kernel_size']
    width = math.floor(kernel_size[0] / 2)
    return np.pad(array, ((0, 0), (width, width), (width, width), (0, 0))) # pad along axis 1 & 2
