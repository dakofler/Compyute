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

        # init if
        w_if = uniform((h_channels, in_channels), -k, k)
        self.w_if = Parameter(w_if, dtype=dtype, label="w_if")
        self.b_if = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_if") if bias else None
        )

        # init hf
        w_hf = uniform((h_channels, h_channels), -k, k)
        self.w_hf = Parameter(w_hf, dtype=dtype, label="w_hf")
        self.b_hf = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_hf") if bias else None
        )

        # init ii
        w_ii = uniform((h_channels, in_channels), -k, k)
        self.w_ii = Parameter(w_ii, dtype=dtype, label="w_ii")
        self.b_ii = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_ii") if bias else None
        )

        # init hi
        w_hi = uniform((h_channels, h_channels), -k, k)
        self.w_hi = Parameter(w_hi, dtype=dtype, label="w_hi")
        self.b_hi = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_hi") if bias else None
        )

        # init ig
        w_ig = uniform((h_channels, in_channels), -k, k)
        self.w_ig = Parameter(w_ig, dtype=dtype, label="w_ig")
        self.b_ig = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_ig") if bias else None
        )

        # init hg
        w_hg = uniform((h_channels, h_channels), -k, k)
        self.w_hg = Parameter(w_hg, dtype=dtype, label="w_hg")
        self.b_hg = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_hg") if bias else None
        )

        # init io
        w_io = uniform((h_channels, in_channels), -k, k)
        self.w_io = Parameter(w_io, dtype=dtype, label="w_io")
        self.b_io = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_io") if bias else None
        )

        # init ho
        w_ho = uniform((h_channels, h_channels), -k, k)
        self.w_ho = Parameter(w_ho, dtype=dtype, label="w_ho")
        self.b_ho = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_ho") if bias else None
        )

    def forward(self, x: Tensor):
        self.check_dims(x, [3])
        x = x.astype(self.dtype)

        # input projections
        # (B, T, Cin) @ (Cin, Ch) + (Ch,) -> (B, T, Ch)
        f_h = linear(x, self.w_if, self.b_if)
        i_h = linear(x, self.w_ii, self.b_ii)
        g_h = linear(x, self.w_ig, self.b_ig)
        o_h = linear(x, self.w_io, self.b_io)

        # iterate over timesteps
        # (B, T, Ch)
        f = zeros_like(f_h)
        i = zeros_like(i_h)
        g = zeros_like(g_h)
        o = zeros_like(o_h)
        c = zeros_like(f_h)
        h = zeros_like(f_h)

        for t in range(x.shape[1]):
            # (B, 4*Ch) + (B, Ch) @ (Ch, 4*Ch) + (4*Ch,) -> (B, 4*Ch)
            f[:, t] = sigmoid(f_h[:, t] + linear(h[:, t - 1], self.w_hf, self.b_hf))
            i[:, t] = sigmoid(i_h[:, t] + linear(h[:, t - 1], self.w_hi, self.b_hi))
            g[:, t] = (g_h[:, t] + linear(h[:, t - 1], self.w_hg, self.b_hg)).tanh()
            o[:, t] = sigmoid(o_h[:, t] + linear(h[:, t - 1], self.w_ho, self.b_ho))

            c[:, t] = f[:, t] * c[:, t - 1] + i[:, t] * g[:, t]
            h[:, t] = o[:, t] * c[:, t].tanh()

        if self.training:

            def backward(dy: Tensor) -> Tensor:
                dh = dy.astype(self.dtype)
                self.set_dy(dh)

                dc = zeros_like(c)

                d_sig_f = zeros_like(f)
                d_sig_i = zeros_like(i)
                d_tanh_g = zeros_like(g)
                d_sig_o = zeros_like(o)

                self.w_hf.grad = zeros_like(self.w_hf)
                self.w_hi.grad = zeros_like(self.w_hi)
                self.w_hg.grad = zeros_like(self.w_hg)
                self.w_ho.grad = zeros_like(self.w_ho)

                for t in range(x.shape[1] - 1, -1, -1):
                    out_grad = dh[:, t]
                    if t < x.shape[1] - 1:
                        out_grad += d_sig_f[:, t + 1] @ self.w_hf
                        out_grad += d_sig_i[:, t + 1] @ self.w_hi
                        out_grad += d_tanh_g[:, t + 1] @ self.w_hg
                        out_grad += d_sig_o[:, t + 1] @ self.w_ho

                    do_t = c[:, t].tanh() * out_grad

                    dc[:, t] = (1 - c[:, t].tanh() ** 2) * o[:, t] * out_grad
                    if t < x.shape[1] - 1:
                        dc[:, t] += f[:, t + 1] * dc[:, t + 1]

                    df_t = (c[:, t - 1] * dc[:, t]) if t > 0 else 0
                    di_t = g[:, t] * dc[:, t]
                    dg_t = i[:, t] * dc[:, t]

                    d_sig_f[:, t] = f[:, t] * (1 - f[:, t]) * df_t
                    d_sig_i[:, t] = i[:, t] * (1 - i[:, t]) * di_t
                    d_tanh_g[:, t] = (1 - g[:, t] ** 2) * dg_t
                    d_sig_o[:, t] = o[:, t] * (1 - o[:, t]) * do_t

                    if t > 0:
                        self.w_hf.grad += d_sig_f[:, t].T @ h[:, t - 1]
                        self.w_hi.grad += d_sig_i[:, t].T @ h[:, t - 1]
                        self.w_hg.grad += d_tanh_g[:, t].T @ h[:, t - 1]
                        self.w_ho.grad += d_sig_o[:, t].T @ h[:, t - 1]

                self.b_hf.grad = d_sig_f.sum(axis=(0, 1))
                self.b_hi.grad = d_sig_i.sum(axis=(0, 1))
                self.b_hg.grad = d_tanh_g.sum(axis=(0, 1))
                self.b_ho.grad = d_sig_o.sum(axis=(0, 1))

                dx = linear_backward(d_sig_f, x, self.w_if, self.b_if)
                dx += linear_backward(d_sig_i, x, self.w_ii, self.b_ii)
                dx += linear_backward(d_tanh_g, x, self.w_ig, self.b_ig)
                dx += linear_backward(d_sig_o, x, self.w_io, self.b_io)

                return dx

            self.backward = backward

        self.set_y(h)
        return h
