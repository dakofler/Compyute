"""Legacy neural network linear functions."""

from typing import Callable, Optional

from ...tensor_ops.transforming import einsum, exp, maximum
from ...tensor_ops.transforming import tanh as cptanh
from ...tensors import Tensor


def linear(
    x: Tensor, w: Tensor, b: Optional[Tensor] = None, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable]]:
    """Applies a linear transformation to the input."""
    y = x @ w.T
    if b:
        y += b

    if return_grad_fn:
        batch_dims = "uvxyz"[: x.n_axes - 1]

        def grad_fn(dy: Tensor) -> tuple[Tensor, Tensor, Optional[Tensor]]:
            # input grads
            dx = dy @ w

            # weight grads, equivalent to dy.T @ x and summing over all batch dims
            dw = einsum(f"{batch_dims}o,{batch_dims}i->oi", dy, x)

            # bias grads, equivalent to summing over all batch dims
            db = einsum(f"{batch_dims}o->o", dy) if b else None

            return dx, dw, db

        return y, grad_fn

    return y, None


def relu(x: Tensor, return_grad_fn: bool = False) -> tuple[Tensor, Optional[Callable]]:
    """Applies the Rectified Linear Unit activation function to an input tensor."""
    y = maximum(x, 0)

    if return_grad_fn:
        return y, (lambda dy: (y > 0) * dy)
    return y, None


def tanh(x: Tensor, return_grad_fn: bool = False) -> tuple[Tensor, Optional[Callable]]:
    """Applies the hyperbolic tangent activationfunction to an input tensor."""
    y = cptanh(x)

    if return_grad_fn:
        return y, (lambda dy: (1 - y**2) * dy)

    return y, None


def sigmoid(
    x: Tensor, return_grad_fn: bool = False
) -> tuple[Tensor, Optional[Callable]]:
    """Applies the sigmoid function to an input tensor."""
    x_exp = exp(x)
    y = x_exp / (1 + x_exp)

    if return_grad_fn:
        return y, (lambda dy: (y * (1 - y)) * dy)

    return y, None
