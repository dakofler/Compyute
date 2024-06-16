"""Neural network functions module"""

from typing import Callable, Optional

from ...base_tensor import Tensor
from ...tensor_functions.computing import tensorprod
from ...tensor_functions.reshaping import reshape, squeeze
from ...tensor_functions.transforming import mean as _mean
from ...tensor_functions.transforming import sum as _sum
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
    return_grad_func: bool = False,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Optional[Callable[[Tensor], tuple[Tensor, Optional[Tensor], Optional[Tensor]]]],
]:
    """Performs 1D batch normalization on a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    rmean: Tensor
        Running mean tensor.
    rvar: Tensor
        Running variance tensor.
    w: Tensor
        Weight tensor for scaling the distribution.
    b: Tensor
        Bias tensor for shifting the distribution.
    m : float, optional
        Momentum used for running mean and variance computation, by default 0.1.
    eps : float, optional
        Constant for numerical stability, by default 1e-5.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

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

    if return_grad_func:
        mean = _mean(x, axis=axes, keepdims=True)
        var = _var(x, axis=axes, keepdims=True)
        var_h = (var + eps) ** -0.5
        x_h = (x - mean) * var_h

        # keep running stats
        rmean *= 1 - m
        rmean += squeeze(mean) * m
        rvar *= 1 - m
        rvar += squeeze(_var(x, axis=axes, keepdims=True, ddof=1)) * m
    else:
        rvar_ = rvar if dim2 else reshape(rvar, shape=(*rvar.shape, 1))
        rmean_ = rmean if dim2 else reshape(rmean, shape=(*rmean.shape, 1))
        var_h = (rvar_ + eps) ** -0.5
        x_h = (x - rmean_) * var_h

    weights = w if dim2 else reshape(w, shape=(*w.shape, 1))
    biases = b if dim2 else reshape(b, shape=(*b.shape, 1))
    y = weights * x_h + biases

    if return_grad_func:

        def grad_func(dy: Tensor) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
            # input grads
            n = tensorprod(x.shape) / x.shape[1]
            dx = (
                weights
                * var_h
                / n
                * (
                    n * dy
                    - _sum(dy, axis=axes, keepdims=True)
                    - x_h * _sum(dy * x_h, axis=axes, keepdims=True)
                )
            )

            # gamma grads
            dw = _sum(x_h * dy, axis=axes) if w.requires_grad else None

            # beta grads
            db = _sum(dy, axis=axes) if b.requires_grad else None

            return dx, dw, db

        return y, rmean, rvar, grad_func

    return y, rmean, rvar, None


def batchnorm2d(
    x: Tensor,
    rmean: Tensor,
    rvar: Tensor,
    w: Tensor,
    b: Tensor,
    m: float = 0.1,
    eps: float = 1e-5,
    return_grad_func: bool = False,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Optional[Callable[[Tensor], tuple[Tensor, Optional[Tensor], Optional[Tensor]]]],
]:
    """Performs 2D batch normalization on a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    rmean: Tensor
        Running mean values.
    rvar: Tensor
        Running variance values.
    w: Tensor
        Weight tensor for scaling the distribution.
    b: Tensor
        Bias tensor for shifting the distribution.
    m : float, optional
        Momentum used for running mean and variance computation, by default 0.1.
    eps : float, optional
        Constant for numerical stability, by default 1e-5.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

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

    if return_grad_func:
        mean = _mean(x, axis=axes, keepdims=True)
        var = _var(x, axis=axes, keepdims=True)
        var_h = (var + eps) ** -0.5
        x_h = (x - mean) * var_h

        # keep running stats
        rmean *= 1 - m
        rmean += squeeze(mean) * m
        rvar *= 1 - m
        rvar += squeeze(_var(x, axis=axes, keepdims=True, ddof=1)) * m
    else:
        rvar_ = reshape(rvar, shape=(*rvar.shape, 1, 1))
        rmean_ = reshape(rmean, shape=(*rmean.shape, 1, 1))
        var_h = (rvar_ + eps) ** -0.5
        x_h = (x - rmean_) * var_h

    weights = reshape(w, shape=(*w.shape, 1, 1))
    biases = reshape(b, shape=(*b.shape, 1, 1))
    y = weights * x_h + biases

    if return_grad_func:

        def grad_func(dy: Tensor) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
            # input grads
            n = tensorprod(x.shape) / x.shape[1]
            dx = (
                weights
                * var_h
                / n
                * (
                    n * dy
                    - _sum(dy, axis=axes, keepdims=True)
                    - x_h * _sum(dy * x_h, axis=axes, keepdims=True)
                )
            )

            # gamma grads
            dw = _sum(x_h * dy, axis=axes) if w.requires_grad else None

            # beta grads
            db = _sum(dy, axis=axes) if b.requires_grad else None

            return dx, dw, db

        return y, rmean, rvar, grad_func

    return y, rmean, rvar, None


def layernorm(
    x: Tensor,
    w: Tensor,
    b: Tensor,
    eps: float = 1e-5,
    return_grad_func: bool = False,
) -> tuple[
    Tensor,
    Optional[Callable[[Tensor], tuple[Tensor, Optional[Tensor], Optional[Tensor]]]],
]:
    """Performs layer normalization on a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w: Tensor
        Weight tensor for scaling the distribution.
    b: Tensor
        Bias tensor for shifting the distribution.
    eps : float, optional
        Constant for numerical stability, by default 1e-5.
    return_grad_func: bool, optional
        Whether to also return the according gradient function, by default False.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], tuple[Tensor, Optional[Tensor], Optional[Tensor]]]], optional
        Gradient function.
    """
    axes = tuple([i for i in range(1, x.ndim)])
    var_h = (_var(x, axis=axes, keepdims=True) + eps) ** -0.5
    x_h = (x - _mean(x, axis=axes, keepdims=True)) * var_h
    y = w * x_h + b

    if return_grad_func:

        def grad_func(dy: Tensor) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
            # input grads
            n = tensorprod(x.shape[1:])
            dx = (
                w
                * var_h
                / n
                * (
                    n * dy
                    - _sum(dy, axis=axes, keepdims=True)
                    - x_h * _sum(dy * x_h, axis=axes, keepdims=True)
                )
            )

            # gamma grads
            dw = _sum(x_h * dy, axis=0) if w.requires_grad else None

            # beta grads
            db = _sum(dy, axis=0) if b.requires_grad else None

            return dx, dw, db

        return y, grad_func

    return y, None
