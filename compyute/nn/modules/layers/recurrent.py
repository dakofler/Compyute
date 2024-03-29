"""Recurrent cells module"""

from ..module import Module
from ...funcional import linear, linear_backward, sigmoid
from ...parameter import Parameter
from ....functional import zeros, zeros_like
from ....random import uniform
from ....tensor import Tensor
from ....types import DtypeLike


__all__ = ["RecurrentCell"]


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
        # (Cin, Ch)
        w_i = uniform((h_channels, in_channels), -k, k)
        self.w_i = Parameter(w_i, dtype=dtype, label="w_i")

        # init input biases
        # (Ch,)
        self.b_i = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_i") if bias else None
        )

        # init hidden weights
        # (Ch, Ch)
        w_h = uniform((h_channels, h_channels), -k, k)
        self.w_h = Parameter(w_h, dtype=dtype, label="w_h")

        # init hidden biases
        # (Ch,)
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
        h = zeros_like(x_h, dtype=self.dtype, device=self.device)
        for t in range(x_h.shape[1]):
            # (B, Ch) @ (Ch, Ch) -> (B, Ch)
            h[:, t] = (x_h[:, t] + linear(h[:, t - 1], self.w_h, self.b_h)).tanh()

        if self.training:

            def backward(dy: Tensor) -> Tensor:
                dh = dy.astype(self.dtype)
                self.set_dy(dh)

                dx_h = zeros_like(x_h, dtype=self.dtype, device=self.device)
                self.w_h.grad = zeros_like(
                    self.w_h, dtype=self.dtype, device=self.device
                )

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

        # input gate
        w_i = uniform((h_channels, in_channels), -k, k)
        self.w_i = Parameter(w_i, dtype=dtype, label="w_i")
        u_i = uniform((h_channels, h_channels), -k, k)
        self.u_i = Parameter(u_i, dtype=dtype, label="u_i")
        self.b_i = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_i") if bias else None
        )

        # forget gate
        w_f = uniform((h_channels, in_channels), -k, k)
        self.w_f = Parameter(w_f, dtype=dtype, label="w_f")
        u_f = uniform((h_channels, h_channels), -k, k)
        self.u_f = Parameter(u_f, dtype=dtype, label="u_f")
        self.b_f = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_f") if bias else None
        )

        # output gate
        w_o = uniform((h_channels, in_channels), -k, k)
        self.w_o = Parameter(w_o, dtype=dtype, label="w_o")
        u_o = uniform((h_channels, h_channels), -k, k)
        self.u_o = Parameter(u_o, dtype=dtype, label="u_o")
        self.b_o = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_o") if bias else None
        )

        # cell
        w_c = uniform((h_channels, in_channels), -k, k)
        self.w_c = Parameter(w_c, dtype=dtype, label="w_c")
        u_c = uniform((h_channels, h_channels), -k, k)
        self.u_c = Parameter(u_c, dtype=dtype, label="u_c")
        self.b_c = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_c") if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [3])
        x = x.astype(self.dtype)

        # input projections
        # (B, T, Cin) @ (Cin, Ch) + (Ch,) -> (B, T, Ch)
        i_h = linear(x, self.w_i.T, self.b_i)
        f_h = linear(x, self.w_f.T, self.b_f)
        o_h = linear(x, self.w_o.T, self.b_o)
        c_h = linear(x, self.w_c.T, self.b_c)

        # iterate over timesteps
        i = zeros_like(i_h, dtype=self.dtype, device=self.device)
        f = zeros_like(f_h, dtype=self.dtype, device=self.device)
        o = zeros_like(o_h, dtype=self.dtype, device=self.device)
        c = zeros_like(c_h, dtype=self.dtype, device=self.device)
        h = zeros_like(c_h, dtype=self.dtype, device=self.device)

        for t in range(x.shape[1]):
            i[:, t] = sigmoid(linear(h[:, t - 1], self.u_i, i_h[:, t]))
            f[:, t] = sigmoid(linear(h[:, t - 1], self.u_f, f_h[:, t]))
            o[:, t] = sigmoid(linear(h[:, t - 1], self.u_o, o_h[:, t]))

            c_t_p = linear(c[:, t - 1], self.u_c, c_h[:, t]).tanh()

            c[:, t] = f[:, t] * c[:, t - 1] + i[:, t] * c_t_p
            h[:, t] = o[:, t] * c[:, t].tanh()

        self.set_y(o)
        return o
