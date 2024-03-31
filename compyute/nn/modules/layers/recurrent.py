"""Recurrent cells module"""

from ..module import Module
from ...funcional import linear, linear_backward, sigmoid
from ...parameter import Parameter
from ....functional import zeros, zeros_like
from ....random import uniform
from ....tensor import Tensor
from ....types import DtypeLike


__all__ = ["LSTMCell", "RecurrentCell"]


class RecurrentCell(Module):
    """Recurrent cell."""

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        dtype: DtypeLike = "float32",
    ) -> None:
        """Recurrent cell.
        Input: (B, T, Cin)
            B ... batch, T ... time, Cin ... input channels
        Output: (B, T, Ch)
            B ... batch, T ... time, Ch ... hidden channels

        Parameters
        ----------
        in_channels : int
            Number of input features.
        h_channels : int
            Number of hidden channels.
        bias : bool, optional
            Whether to use bias values, by default True.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.bias = bias
        self.dtype = dtype

        k = h_channels**-0.5

        # init input weights
        w_i = uniform((h_channels, in_channels), -k, k)
        self.w_i = Parameter(w_i, dtype=dtype, label="w_i")

        # init input biases
        self.b_i = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_i") if bias else None
        )

        # init hidden weights
        w_h = uniform((h_channels, h_channels), -k, k)
        self.w_h = Parameter(w_h, dtype=dtype, label="w_h")

        # init hidden biases
        self.b_h = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_h") if bias else None
        )

    def __repr__(self):
        name = self.__class__.__name__
        in_channels = self.in_channels
        h_channels = self.h_channels
        bias = self.bias
        dtype = self.dtype
        return f"{name}({in_channels=}, {h_channels=}, {bias=}, {dtype=})"

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [3])
        x = x.astype(self.dtype)

        # input projection
        # (B, T, Cin) @ (Cin, Ch)  -> (B, T, Ch)
        x_h = linear(x, self.w_i, self.b_i)

        # iterate over timesteps
        h = zeros_like(x_h)
        for t in range(x_h.shape[1]):
            # (B, Ch) @ (Ch, Ch) -> (B, Ch)
            h[:, t] = (x_h[:, t] + linear(h[:, t - 1], self.w_h, self.b_h)).tanh()

        if self.training:

            def backward(dy: Tensor) -> Tensor:
                dh = dy.astype(self.dtype)
                self.set_dy(dh)

                dx_h = zeros_like(x_h)
                self.w_h.grad = zeros_like(self.w_h)

                for t in range(x.shape[1] - 1, -1, -1):
                    # add hidden state grad of next t, if not last t
                    if t == x_h.shape[1] - 1:
                        out_grad = dh[:, t]
                    else:
                        # (B, Ch) @ (Ch, Ch) + (B, Ch) -> (B, Ch)
                        out_grad = dx_h[:, t + 1] @ self.w_h + dh[:, t]

                    # activation grads
                    dx_h[:, t] = (1 - h[:, t] ** 2) * out_grad

                    # hidden weight grads
                    # (Ch, B) @ (B, Ch) -> (Ch, Ch)
                    if t > 0:
                        self.w_h.grad += dx_h[:, t].T @ h[:, t - 1]

                # hidden bias grads
                # (B, T, Ch) -> (Ch,)
                self.b_h.grad = dx_h.sum((0, 1))

                # input grads
                return linear_backward(dx_h, x, self.w_i, self.b_i)

            self.backward = backward

        self.set_y(h)
        return h


class LSTMCell(Module):
    """Long Short-Term Memory cell."""

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        dtype: DtypeLike = "float32",
    ) -> None:
        """Long Short-Term Memory cell.
        Input: (B, T, Cin)
            B ... batch, T ... time, Cin ... input channels
        Output: (B, Ch)
            B ... batch, Ch ... hidden channels

        Parameters
        ----------
        in_channels : int
            Number of input features.
        h_channels : int
            Number of hidden channels.
        bias : bool, optional
            Whether to use bias values, by default True.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.bias = bias
        self.dtype = dtype

        k = in_channels**-0.5

        # init i
        w_i = uniform((4 * h_channels, in_channels), -k, k)
        self.w_i = Parameter(w_i, dtype=dtype, label="w_i")
        self.b_i = (
            Parameter(zeros((4 * h_channels,)), dtype=dtype, label="b_i")
            if bias
            else None
        )

        # init h
        w_h = uniform((4 * h_channels, h_channels), -k, k)
        self.w_h = Parameter(w_h, dtype=dtype, label="w_h")
        self.b_h = (
            Parameter(zeros((4 * h_channels,)), dtype=dtype, label="b_h")
            if bias
            else None
        )

    def __repr__(self):
        name = self.__class__.__name__
        in_channels = self.in_channels
        h_channels = self.h_channels
        bias = self.bias
        dtype = self.dtype
        return f"{name}({in_channels=}, {h_channels=}, {bias=}, {dtype=})"

    def forward(self, x: Tensor):
        self.check_dims(x, [3])
        x = x.astype(self.dtype)
        n = self.h_channels

        # input projections
        # (B, T, Cin) @ (Cin, 4*Ch) + (4*Ch,) -> (B, T, 4*Ch)
        x_h = linear(x, self.w_i, self.b_i)

        # iterate over timesteps
        ifgo = zeros_like(x_h)
        c = zeros_like(x_h[:, :, :n])
        h = zeros_like(c)

        for t in range(x.shape[1]):
            # (B, 4*Ch) + (B, Ch) @ (Ch, 4*Ch) + (4*Ch,) -> (B, 4*Ch)
            ifgo_preact = x_h[:, t] + linear(h[:, t - 1], self.w_h, self.b_h)

            ifgo[:, t, : 2 * n] = sigmoid(ifgo_preact[:, : 2 * n])
            ifgo[:, t, 2 * n : 3 * n] = ifgo_preact[:, 2 * n : 3 * n].tanh()
            ifgo[:, t, -n:] = sigmoid(ifgo_preact[:, -n:])

            c[:, t] = (
                ifgo[:, t, n : 2 * n] * c[:, t - 1]
                + ifgo[:, t, :n] * ifgo[:, t, 2 * n : 3 * n]
            )
            h[:, t] = ifgo[:, t, -n:] * c[:, t].tanh()

        if self.training:

            # https://pureai.com/articles/2019/11/14/~/media/ECG/pureai/Images/2019/11/lstm_1.asxh

            def backward(dy: Tensor) -> Tensor:
                dh = dy.astype(self.dtype)
                self.set_dy(dh)

                dc = zeros_like(c)
                difgo_preact = zeros_like(ifgo)
                self.w_h.grad = zeros_like(self.w_h)

                for t in range(x.shape[1] - 1, -1, -1):
                    difgo_t = zeros_like(ifgo[:, 1])

                    out_grad = dh[:, t]
                    if t < x.shape[1] - 1:
                        out_grad += difgo_preact[:, t + 1] @ self.w_h

                    dc[:, t] = (1 - c[:, t].tanh() ** 2) * ifgo[:, t, -n:] * out_grad
                    if t < x.shape[1] - 1:
                        dc[:, t] += ifgo[:, t + 1, n : 2 * n] * dc[:, t + 1]

                    difgo_t[:, :n] = ifgo[:, t, 2 * n : 3 * n] * dc[:, t]
                    difgo_t[:, n : 2 * n] = (c[:, t - 1] * dc[:, t]) if t > 0 else 0
                    difgo_t[:, 2 * n : 3 * n] = ifgo[:, t, :n] * dc[:, t]
                    difgo_t[:, -n:] = c[:, t].tanh() * out_grad

                    difgo_preact[:, t, : 2 * n] = (
                        ifgo[:, t, : 2 * n]
                        * (1 - ifgo[:, t, : 2 * n])
                        * difgo_t[:, : 2 * n]
                    )
                    difgo_preact[:, t, 2 * n : 3 * n] = (
                        1 - ifgo[:, t, 2 * n : 3 * n] ** 2
                    ) * difgo_t[:, 2 * n : 3 * n]
                    difgo_preact[:, t, -n:] = (
                        ifgo[:, t, -n:] * (1 - ifgo[:, t, -n:]) * difgo_t[:, -n:]
                    )

                    if t > 0:
                        self.w_h.grad += difgo_preact[:, t].T @ h[:, t - 1]

                self.b_h.grad = difgo_preact.sum(axis=(0, 1))

                return linear_backward(difgo_preact, x, self.w_i, self.b_i)

            self.backward = backward

        self.set_y(h)
        return h
