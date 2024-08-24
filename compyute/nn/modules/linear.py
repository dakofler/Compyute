"""Neural network linear transformation modules."""

import math
from typing import Optional

from ...random.random import uniform
from ...tensor_ops.creating import zeros
from ...tensors import Tensor
from ...typing import DType, float32
from ..functional.linear import linear
from ..parameter import Parameter, update_parameter_grad
from .module import Module

__all__ = ["Linear"]


class Linear(Module):
    r"""Applies a linear transformation to the input.

    .. math::
        y = xW^T + b

    Shapes:
        - Input :math:`(B_1, ... , B_n, C_{in})`
        - Output :math:`(B_1, ... , B_n, C_{out})`
    where
        - :math:`B_1, ... , B_n` ... batch axes
        - :math:`C_{in}` ... input channels
        - :math:`C_{out}` ... output channels

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (neurons).
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    dtype : DType, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in}}}`. Biases are initialized as zeros.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        dtype: DType = float32,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        # init weights
        k = 1 / math.sqrt(in_channels)
        self.w = Parameter(uniform((out_channels, in_channels), -k, k, dtype=dtype))

        # init biases
        self.b = Parameter(zeros((out_channels,), dtype=dtype)) if bias else None

    def forward(self, x: Tensor) -> Tensor:

        y, grad_fn = linear(x, self.w, self.b, self._is_training)

        if self._is_training and grad_fn is not None:

            def _backward(dy: Tensor) -> Tensor:
                dx, dw, db = grad_fn(dy)
                update_parameter_grad(self.w, dw)
                update_parameter_grad(self.b, db)
                return dx

            self._backward = _backward

        return y
