"""Neural network linear transformation modules."""

from typing import Optional

from ...base_tensor import Tensor
from ...dtypes import Dtype, _DtypeLike
from ...random.random import uniform
from ...tensor_functions.creating import zeros
from ..functional.linear import linear
from ..parameter import Parameter
from .module import Module

__all__ = ["Linear"]


class Linear(Module):
    """Fully connected layer. Applies a linear transformation to the incoming data.

    Input: (B, ... , Cin)
        B ... batch, Cin ... input channels
    Output: (B, ... , Co)
        B ... batch, Co ... output channels

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (neurons).
    bias : bool, optional
        Whether to use bias values, by default True.
    dtype : DtypeLike, optional
        Datatype of weights and biases, by default Dtype.FLOAT32.
    label : str, optional
        Module label.
    training : bool, optional
        Whether the module should be in training mode, by default False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        super().__init__(label, training)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.dtype = Dtype(dtype)

        # init weights
        k = in_channels**-0.5
        self.w = Parameter(uniform((out_channels, in_channels), -k, k, dtype), label="lin_w")

        # init biases
        self.b = Parameter(zeros((out_channels,), dtype), label="lin_b") if bias else None

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [2, 3, 4, 5])
        x = x.as_type(self.dtype)
        y, grad_fn = linear(x, self.w, self.b, self._training)

        if self._training and grad_fn is not None:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.as_type(self.dtype)
                dx, dw, db = grad_fn(dy)
                self._update_parameter_grad(self.w, dw)
                self._update_parameter_grad(self.b, db)
                return dx

            self._backward = _backward

        return y
