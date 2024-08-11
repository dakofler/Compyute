"""Neural network normalization functions."""

import math
from typing import Callable, Optional

from ...base_tensor import Tensor
from ...tensor_ops.reshaping import reshape, squeeze
from ...tensor_ops.transforming import mean as cpmean
from ...tensor_ops.transforming import norm, sqrt
from ...tensor_ops.transforming import sum as cpsum
from ...tensor_ops.transforming import var as cpvar

__all__ = ["batchnorm1d", "batchnorm2d", "layernorm", "rmsnorm"]


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

    See Also
    ----------
    :class:`compyute.nn.BatchNorm1D`
    """
    x_is_2d = x.n_axes == 2
    axes = 0 if x_is_2d else (0, 2)

    if return_grad_fn:
        # compute mean and variance from x
        mean = cpmean(x, axis=axes, keepdims=True)
        var = cpvar(x, axis=axes, keepdims=True)
        inv_std = (var + eps) ** -0.5
        x_norm = (x - mean) * inv_std

        # keep running stats
        rmean = rmean * (1 - m) + squeeze(mean) * m
        rvar = rvar * (1 - m) + cpvar(x, axis=axes, ddof=1) * m
    else:
        # use running mean and variance
        rvar_ = rvar if x_is_2d else rvar.to_shape((*rvar.shape, 1))
        rmean_ = rmean if x_is_2d else rmean.to_shape((*rmean.shape, 1))
        inv_std = (rvar_ + eps) ** -0.5
        x_norm = (x - rmean_) * inv_std

    w = w if x_is_2d else reshape(w, shape=(*w.shape, 1))
    b = b if x_is_2d else reshape(b, shape=(*b.shape, 1))
    y = w * x_norm + b

    if return_grad_fn:

        def grad_fn(dy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            # input grads
            n = x.size / x.shape[1]

            dy_sum = cpsum(dy, axis=axes, keepdims=True)
            dy_x_norm_sum = cpsum(dy * x_norm, axis=axes, keepdims=True)
            dx = w * inv_std / n * (n * dy - dy_sum - x_norm * dy_x_norm_sum)

            # gamma grads
            dw = squeeze(dy_x_norm_sum)

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

    See Also
    ----------
    :class:`compyute.nn.BatchNorm2D`
    """
    axes = (0, 2, 3)

    if return_grad_fn:
        # compute mean and variance from x
        mean = cpmean(x, axis=axes, keepdims=True)
        var = cpvar(x, axis=axes, keepdims=True)
        inv_std = (var + eps) ** -0.5
        x_norm = (x - mean) * inv_std

        rmean = rmean * (1 - m) + squeeze(mean) * m
        rvar = rvar * (1 - m) + cpvar(x, axis=axes, ddof=1) * m
    else:
        # use running mean and variance
        inv_std = (rvar.to_shape((*rvar.shape, 1, 1)) + eps) ** -0.5
        x_norm = (x - rmean.to_shape((*rmean.shape, 1, 1))) * inv_std

    w = w.to_shape((*w.shape, 1, 1))
    b = b.to_shape((*b.shape, 1, 1))
    y = w * x_norm + b

    if return_grad_fn:

        def grad_fn(dy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            # input grads
            n = x.size / x.shape[1]

            dy_sum = cpsum(dy, axis=axes, keepdims=True)
            dy_x_norm_sum = cpsum(dy * x_norm, axis=axes, keepdims=True)
            dx = w * inv_std / n * (n * dy - dy_sum - x_norm * dy_x_norm_sum)

            # gamma grads
            dw = squeeze(dy_x_norm_sum)

            # beta grads
            db = squeeze(dy_sum)

            return dx, dw, db

        return y, rmean, rvar, grad_fn

    return y, rmean, rvar, None


def layernorm(x: Tensor, w: Tensor, b: Tensor, eps: float = 1e-5, return_grad_fn: bool = False) -> tuple[
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

    See Also
    ----------
    :class:`compyute.nn.LayerNorm`
    """
    axes = tuple(-i - 1 for i in range(w.n_axes))
    inv_std = (cpvar(x, axis=axes, keepdims=True) + eps) ** -0.5
    x_norm = (x - cpmean(x, axis=axes, keepdims=True)) * inv_std
    y = w * x_norm + b

    if return_grad_fn:
        sum_axes = tuple(range(x.n_axes - w.n_axes))

        def grad_fn(dy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            # input grads
            dy_sum = cpsum(dy, axis=axes, keepdims=True)
            dy_x_std_sum = cpsum(dy * x_norm, axis=axes, keepdims=True)
            dx = w * inv_std / w.size * (w.size * dy - dy_sum - x_norm * dy_x_std_sum)

            # gamma grads
            dw = cpsum(dy * x_norm, axis=sum_axes)

            # beta grads
            db = cpsum(dy, axis=sum_axes)

            return dx, dw, db

        return y, grad_fn

    return y, None


def rmsnorm(x: Tensor, w: Tensor, eps: float = 1e-5, return_grad_fn: bool = False) -> tuple[
    Tensor,
    Optional[Callable[[Tensor], tuple[Tensor, Tensor]]],
]:
    """Performs RMS normalization on a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w : Tensor
        Weight tensor for scaling the distribution.
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

    See Also
    ----------
    :class:`compyute.nn.RMSNorm`
    """
    axes = tuple(-i - 1 for i in range(w.n_axes))
    rms = (cpmean(x**2, axis=axes, keepdims=True) + eps) ** 0.5
    x_norm = x / rms
    y = w * x_norm

    if return_grad_fn:
        sum_axes = tuple(range(x.n_axes - w.n_axes))

        def grad_fn(dy: Tensor) -> tuple[Tensor, Tensor]:
            # input grads
            dx = w * (dy / rms - x * cpsum(dy * x, axis=axes, keepdims=True) / (w.size * rms**3))

            # gamma grads
            dw = cpsum(dy * x_norm, axis=sum_axes)

            return dx, dw

        return y, grad_fn

    return y, None
