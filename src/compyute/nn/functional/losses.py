"""Neural network functions module"""

from typing import Callable, Optional

from ...base_tensor import Tensor
from ...preprocessing.basic import one_hot_encode
from ...tensor_functions.computing import tensorprod
from ...tensor_functions.transforming import clip, log, mean
from ...tensor_functions.transforming import sum as _sum
from .activations import softmax

__all__ = ["mean_squared_error", "cross_entropy", "binary_cross_entropy"]


def mean_squared_error(
    y: Tensor, t: Tensor, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[], Tensor]]]:
    """Computes the mean squared error loss.

    Parameters
    ----------
    y : Tensor
        A model's predictions.
    t : Tensor
        Target values.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Mean squared error loss.
    Callable[[], Tensor]], optional
        Gradient function.
    """
    dif = y.float() - t.float()
    loss = mean(dif**2)

    grad_func = (lambda: dif * 2 / tensorprod(y.shape)) if return_grad_func else None

    return loss, grad_func


def cross_entropy(
    y: Tensor, t: Tensor, eps: float = 1e-8, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[], Tensor]]]:
    """Computes the cross entropy loss.

    Parameters
    ----------
    y : Tensor
        Model logits.
    t : Tensor
        Target integer class labels.
    eps : float, optional
        Constant used for numerical stability, by default 1e-8.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Cross entropy loss.
    Callable[[], Tensor]], optional
        Gradient function.
    """
    probs, _ = softmax(y.float(), False)
    t = one_hot_encode(t.int(), y.shape[-1])
    loss = -mean(log(_sum((probs + eps) * t, axis=-1)))

    grad_func = (lambda: (probs - t) / tensorprod(y.shape[:-1])) if return_grad_func else None

    return loss, grad_func


def binary_cross_entropy(
    y: Tensor, t: Tensor, return_grad_func: bool = False
) -> tuple[Tensor, Optional[Callable[[], Tensor]]]:
    """Computes the cross entropy loss.

    Parameters
    ----------
    y : Tensor
        Model logits.
    t : Tensor
        Binary target values.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Cross entropy loss.
    Callable[[], Tensor]], optional
        Gradient function.
    """
    c = 100
    loss = -mean(t * clip(log(y), -c, c) + (1 - t) * clip(log(1 - y), -c, c))

    grad_func = (
        (lambda: (-t / y + (1 - t) / (1 - y)) / tensorprod(y.shape)) if return_grad_func else None
    )

    return loss, grad_func
