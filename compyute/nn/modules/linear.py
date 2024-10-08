"""Neural network linear transformation modules."""

import math
from typing import Optional

from ...tensor_ops.creation_ops import empty
from ...tensors import Tensor
from ...typing import DType
from ..functional.linear_funcs import LinearFn
from ..parameter import Parameter, update_parameter_grad
from ..utils.initializers import init_uniform
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
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights and biases are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in}}}`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        # init parameters
        self.w = Parameter(empty((out_channels, in_channels), dtype=dtype))
        self.b = Parameter(empty((out_channels,), dtype=dtype)) if bias else None
        self._init_parameters_and_buffers()

    def _init_parameters_and_buffers(self) -> None:
        std = 1.0 / math.sqrt(self.in_channels)
        init_uniform(self.w, low=-std, high=std)
        if self.b:
            init_uniform(self.b, low=-std, high=std)

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return LinearFn.forward(self.fcache, x, self.w, self.b)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw, db = LinearFn.backward(self.fcache, dy)
        update_parameter_grad(self.w, dw)
        update_parameter_grad(self.b, db)
        return dx
