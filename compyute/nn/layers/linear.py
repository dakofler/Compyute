"""parameter layers layer"""

from compyute.functional import arange, random_uniform, zeros
from compyute.nn.module import Module
from compyute.nn.parameter import Parameter
from compyute.tensor import Tensor, ArrayLike


__all__ = ["Linear"]


class Linear(Module):
    """Fully connected layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bias: bool = True,
        dtype: str = "float32",
    ) -> None:
        """Fully connected layer.
        Input: (B, ... , Cin)
            B ... batch, Cin ... input channels
        Output: (B, ... , Co)
            B ... batch, Co ... output channels

        Parameters
        ----------
        in_channels : int
            Number of input channels of the layer.
        out_channels : int
            Number of output channels (neurons) of the layer.
        use_bias : bool, optional
            Whether to use bias values, by default True.
        dtype: str, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.dtype = dtype

        # init weights
        # (Ci, Co)
        k = in_channels**-0.5
        w = random_uniform((in_channels, out_channels), -k, k)
        self.w = Parameter(w, dtype=dtype, label="w")

        # init biases
        # (Co,)
        if use_bias:
            self.b = Parameter(zeros((out_channels,)), dtype=dtype, label="b")

    def __repr__(self) -> str:
        name = self.__class__.__name__
        in_channels = self.in_channels
        out_channels = self.out_channels
        use_bias = self.use_bias
        dtype = self.dtype
        return f"{name}({in_channels=}, {out_channels=}, {use_bias=}, {dtype=})"

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [2, 3])
        x = x.astype(self.dtype)

        # (B, ... , Ci) @ (Ci, Co) -> (B, ... , Co)
        y = x @ self.w

        if self.use_bias:
            # (B, ... , Co) + (Co,)
            y += self.b

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                dy = dy.astype(self.dtype)

                # input grads
                # (B, ... , Co) @ (Co, Ci) -> (B, ..., Ci)
                dx = dy @ self.w.T

                # weight grads
                # 2D: (Ci, B) @ (B, Co) -> (Ci, Co)
                # ND: (B, ..., Ci, Bn) @ (B, ... , Bn, Co) -> (B, ..., Ci, Co)
                dw = x.transpose().data @ dy
                if x.ndim > 2:
                    # sum over all batch dimensions
                    # (B, ..., Ci, Co) -> (Ci, Co)
                    dw = dw.sum(axis=tuple(arange(x.ndim - 2)))

                self.w.grad = dw

                # bias grads
                if self.use_bias:
                    # sum over all batch dimensions
                    # (B, ... , Co) -> (Co,)
                    self.b.grad = dy.sum(axis=tuple(arange(x.ndim - 1)))

                return dx

            self.backward = backward

        self.set_y(y)
        return y
