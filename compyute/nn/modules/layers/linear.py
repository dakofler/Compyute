"""Linear transformation layers module"""

from typing import Optional
from ..module import Module
from ...parameter import Parameter
from ...functional import linear
from ....tensor_f import zeros
from ....random import uniform
from ....tensor import Tensor
from ....types import DtypeLike


__all__ = ["Linear"]


class Linear(Module):
    """Fully connected layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        dtype: DtypeLike = "float32",
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
        self.w = Parameter(uniform((out_channels, in_channels), -k, k), dtype=dtype, label="w")

        # init biases
        # (Co,)
        self.b = Parameter(zeros((out_channels,)), dtype=dtype, label="b") if bias else None

    def __repr__(self) -> str:
        label = self.label
        in_channels = self.in_channels
        out_channels = self.out_channels
        bias = self.bias
        dtype = self.dtype
        return f"{label}({in_channels=}, {out_channels=}, {bias=}, {dtype=})"

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [2, 3, 4, 5])
        x = x.astype(self.dtype)

        y, linear_backward = linear(x, self.w, self.b, self.training)

        if self.training:

            def backward(dy: Tensor) -> Tensor:
                dy = dy.astype(self.dtype)

                # compute gradients
                dx, self.w.grad, db = linear_backward(dy)

                if self.b is not None:
                    self.b.grad = db

                return dx

            self.backward_fn = backward

        return y
