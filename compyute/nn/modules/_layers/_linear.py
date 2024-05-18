"""Linear transformation layers module"""

from typing import Optional

from ...._tensor_functions import zeros
from ...._types import _DtypeLike
from ....random import uniform
from ....tensors import Tensor
from ...functional import linear
from ...parameter import Parameter
from .._module import Module

__all__ = ["Linear"]


class Linear(Module):
    """Fully connected layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        dtype: _DtypeLike = "float32",
        label: Optional[str] = None,
    ) -> None:
        """Fully connected layer.
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
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        label: str, optional
            Module label.
        """
        super().__init__(label)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.dtype = dtype

        # init weights
        # (Co, Ci)
        k = in_channels**-0.5
        self.w = Parameter(uniform((out_channels, in_channels), -k, k, dtype), label="w")

        # init biases
        # (Co,)
        self.b = Parameter(zeros((out_channels,), dtype), label="b") if bias else None

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [2, 3, 4, 5])
        x = x.astype(self.dtype)

        y, linear_backward = linear(x, self.w, self.b, self.training)

        if self.training:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.astype(self.dtype)

                # compute gradients
                dx, self.w.grad, db = linear_backward(dy)

                if self.b is not None:
                    self.b.grad = db

                return dx

            self._backward = _backward

        return y
