"""Neural network normalization functions."""

from functools import reduce
from operator import mul
from typing import Callable, Optional

from ...base_tensor import Tensor
from ...tensor_functions.reshaping import reshape, squeeze
from ...tensor_functions.transforming import mean as _mean
from ...tensor_functions.transforming import sum as cpsum
from ...tensor_functions.transforming import var as _var

__all__ = ["batchnorm1d", "batchnorm2d", "layernorm"]


def batchnorm1d(
    x: Tensor,
    rmean: Tensor,
    rvar: Tensor,
    w: Tensor,
    b: Tensor,
    m: float = 0.1,
    eps: float = 1e-5,
    return_grad_fn: bool = False,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Optional[Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]],
]:
    """Performs 1D batch normalization on a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    rmean : Tensor
        Running mean tensor.
    rvar : Tensor
        Running variance tensor.
    w : Tensor
        Weight tensor for scaling the distribution.
    b : Tensor
        Bias tensor for shifting the distribution.
    m : float, optional
        Momentum used for running mean and variance computation. Defaults to ``0.1``.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-5``.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Tensor
        New running mean.
    Tensor
        New running variance.
    Callable[[Tensor], tuple[Tensor, Optional[Tensor], Optional[Tensor]]]], optional
        Gradient function.
    """
    dim2 = x.ndim == 2
    axes = 0 if dim2 else (0, 2)

    if return_grad_fn:
        mean = _mean(x, axis=axes, keepdims=True)
        var = _var(x, axis=axes, keepdims=True)
        inv_std = (var + eps) ** -0.5
        x_std = (x - mean) * inv_std

        # keep running stats
        rmean *= 1 - m
        rmean += squeeze(mean) * m
        rvar *= 1 - m
        rvar += squeeze(_var(x, axis=axes, keepdims=True, ddof=1)) * m
    else:
        rvar_ = rvar if dim2 else reshape(rvar, shape=(*rvar.shape, 1))
        rmean_ = rmean if dim2 else reshape(rmean, shape=(*rmean.shape, 1))
        inv_std = (rvar_ + eps) ** -0.5
        x_std = (x - rmean_) * inv_std

    weights = w if dim2 else reshape(w, shape=(*w.shape, 1))
    biases = b if dim2 else reshape(b, shape=(*b.shape, 1))
    y = weights * x_std + biases

    if return_grad_fn:

        def grad_fn(dy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            # input grads
            n = reduce(mul, x.shape) / x.shape[1]

            dy_sum = cpsum(dy, axis=axes, keepdims=True)
            dy_x_std_sum = cpsum(dy * x_std, axis=axes, keepdims=True)
            dx = weights * inv_std / n * (n * dy - dy_sum - x_std * dy_x_std_sum)

            # gamma grads
            dw = squeeze(dy_x_std_sum)

            # beta grads
            db = squeeze(dy_sum)

            return dx, dw, db

        return y, rmean, rvar, grad_fn

    return y, rmean, rvar, None


def batchnorm2d(
    x: Tensor,
    rmean: Tensor,
    rvar: Tensor,
    w: Tensor,
    b: Tensor,
    m: float = 0.1,
    eps: float = 1e-5,
    return_grad_fn: bool = False,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Optional[Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]],
]:
    """Performs 2D batch normalization on a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    rmean : Tensor
        Running mean values.
    rvar : Tensor
        Running variance values.
    w : Tensor
        Weight tensor for scaling the distribution.
    b : Tensor
        Bias tensor for shifting the distribution.
    m : float, optional
        Momentum used for running mean and variance computation. Defaults to ``0.1``.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-5``.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Tensor
        New running mean.
    Tensor
        New running variance.
    Callable[[Tensor], tuple[Tensor, Optional[Tensor], Optional[Tensor]]]], optional
        Gradient function.
    """
    axes = (0, 2, 3)

    if return_grad_fn:
        mean = _mean(x, axis=axes, keepdims=True)
        var = _var(x, axis=axes, keepdims=True)
        inv_std = (var + eps) ** -0.5
        x_std = (x - mean) * inv_std

        # keep running stats
        rmean *= 1 - m
        rmean += squeeze(mean) * m
        rvar *= 1 - m
        rvar += squeeze(_var(x, axis=axes, keepdims=True, ddof=1)) * m
    else:
        rvar_ = reshape(rvar, shape=(*rvar.shape, 1, 1))
        rmean_ = reshape(rmean, shape=(*rmean.shape, 1, 1))
        inv_std = (rvar_ + eps) ** -0.5
        x_std = (x - rmean_) * inv_std

    weights = reshape(w, shape=(*w.shape, 1, 1))
    biases = reshape(b, shape=(*b.shape, 1, 1))
    y = weights * x_std + biases

    if return_grad_fn:

        def grad_fn(dy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            # input grads
            n = reduce(mul, x.shape) / x.shape[1]

            dy_sum = cpsum(dy, axis=axes, keepdims=True)
            dy_x_std_sum = cpsum(dy * x_std, axis=axes, keepdims=True)
            dx = weights * inv_std / n * (n * dy - dy_sum - x_std * dy_x_std_sum)

            # gamma grads
            dw = squeeze(dy_x_std_sum)

            # beta grads
            db = squeeze(dy_sum)

            return dx, dw, db

        return y, rmean, rvar, grad_fn

    return y, rmean, rvar, None


def layernorm(
    x: Tensor,
    w: Tensor,
    b: Tensor,
    eps: float = 1e-5,
    return_grad_fn: bool = False,
) -> tuple[
    Tensor,
    Optional[Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]],
]:
    """Performs layer normalization on a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w : Tensor
        Weight tensor for scaling the distribution.
    b : Tensor
        Bias tensor for shifting the distribution.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-5``.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], tuple[Tensor, Optional[Tensor], Optional[Tensor]]]], optional
        Gradient function.
    """
    axes = tuple(-i - 1 for i in range(w.ndim))
    inv_std = (_var(x, axis=axes, keepdims=True) + eps) ** -0.5
    x_std = (x - _mean(x, axis=axes, keepdims=True)) * inv_std
    y = w * x_std + b

    if return_grad_fn:
        sum_axes = tuple(range(x.ndim - w.ndim))

        def grad_fn(dy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            # input grads
            dy_sum = cpsum(dy, axis=axes, keepdims=True)
            dy_x_std_sum = cpsum(dy * x_std, axis=axes, keepdims=True)
            dx = w * inv_std / w.size * (w.size * dy - dy_sum - x_std * dy_x_std_sum)

            # gamma grads
            dw = cpsum(dy * x_std, axis=sum_axes)

            # beta grads
            db = cpsum(dy, axis=sum_axes)

            return dx, dw, db

        return y, grad_fn

    return y, None
