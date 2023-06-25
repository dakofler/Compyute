"""padding functions module"""

import math
import numpy as np

from walnut.tensor import Tensor


def valid(x: Tensor, **kwargs) -> Tensor:
    """Applies valid padding using zero-values to a tensor.

    ### Parameters
        x: `Tensor`
            Tensor, padding is applied to.

    ### Returns
        y: `Tensor`
            Padded tensor.
    """
    return x

def same(x: Tensor, **kwargs) -> Tensor:
    """Applies same padding using zero-values to a tensor.

    ### Parameters
        x: `Tensor`
            Tensor, padding is applied to.
        **kwargs:
            kernel_shape: `tuple[int]`
                Kernel shape the padding width is adapted to.
    ### Returns
        y: `Tensor`
            Padded tensor.
    """
    kernel_shape = kwargs['kernel_shape']
    width = math.floor(kernel_shape[0] / 2)
    return Tensor(np.pad(x.data, ((0, 0), (0, 0), (width, width), (width, width))))
