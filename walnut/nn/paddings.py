"""Padding functions module"""

import math
import numpy as np

from walnut.tensor import Tensor


def valid(x: Tensor, **kwargs) -> Tensor:
    """Applies valid padding using zero-values to a tensor.

    Parameters
    ----------
    x : Tensor
        Tensor, where the padding function is applied to.

    Returns
    -------
    Tensor
        Padded tensor.
    """
    return x


def same(x: Tensor, **kwargs) -> Tensor:
    """Applies same padding using zero-values to a tensor.

    Parameters
    ----------
    x : Tensor
        Tensor, where the padding function is applied to.

    Returns
    -------
    Tensor
        Padded tensor.
    """
    kernel_shape = kwargs["kernel_shape"]
    width = math.floor(kernel_shape[0] / 2)
    return Tensor(np.pad(x.data, ((0, 0), (0, 0), (width, width), (width, width))))
