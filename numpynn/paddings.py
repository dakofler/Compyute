import numpy as np
import math

def Valid(array: np.ndarray, kernel_size: tuple[int, int]=None) -> np.ndarray:
    """Applies Valid padding using zero-values to an input array.

    Args:
        array: Array padding is to be applied to.
        kernel_size: Kernel size the padding width is adapted to [optional].

    Returns:
        Padded array.
    """
    return array.copy()

def Same(array: np.ndarray, kernel_size: tuple[int, int]=None) -> np.ndarray:
    """Applies Same padding using zero-values to an input array.

    Args:
        array: Array padding is to be applied to.
        kernel_size: Kernel size the padding width is adapted to.

    Returns:
        Padded array.
    """
    width = math.floor(kernel_size[0] / 2)
    return np.pad(array, width)[width : -width, :, :, width : -width]