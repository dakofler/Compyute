"""Recurrent cells module"""

from ..module import Module
from ...funcional import linear, linear_backward, sigmoid
from ...parameter import Parameter
from ....functional import zeros, zeros_like
from ....random import uniform
from ....tensor import Tensor
from ....types import DtypeLike


__all__ = ["LSTM", "Recurrent"]


class Recurrent(Module):
    """Recurrent module."""

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        return_sequence: bool = True,
        dtype: DtypeLike = "float32",
    ) -> None:
        """Recurrent module.
        Input: (B, T, Cin)
            B ... batch, T ... time, Cin ... input channels
        Output: (B, T, Ch) if return_sequence=True else (B, Ch)
            B ... batch, T ... time, Ch ... hidden channels

        Parameters
        ----------
        in_channels : int
            Number of input features.
        h_channels : int
            Number of hidden channels.
        bias : bool, optional
            Whether to use bias values, by default True.
        return_sequence: bool, optional
            Whether to return the entire sequence or only the last hidden state.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.bias = bias
        self.return_sequence = return_sequence
        self.dtype = dtype

        k = h_channels**-0.5

        # init input weights and biases
        w_i = uniform((h_channels, in_channels), -k, k)
        self.w_i = Parameter(w_i, dtype=dtype, label="w_i")
        self.b_i = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_i") if bias else None
        )

        # init hidden weights and biases
        w_h = uniform((h_channels, h_channels), -k, k)
        self.w_h = Parameter(w_h, dtype=dtype, label="w_h")
        self.b_h = (
            Parameter(zeros((h_channels,)), dtype=dtype, label="b_h") if bias else None
        )

    def __repr__(self):
        name = self.__class__.__name__
        in_channels = self.in_channels
        h_channels = self.h_channels
        bias = self.bias
        return_sequence = self.return_sequence
        dtype = self.dtype
        return f"{name}({in_channels=}, {h_channels=}, {bias=}, {return_sequence=}, {dtype=})"

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [3])
        x = x.astype(self.dtype)

        # input
        # (B, T, Cin) @ (Cin, Ch) -> (B, T, Ch)
        x_h = linear(x, self.w_i, self.b_i)

        # iterate over timesteps
        h = zeros_like(x_h)
        for t in range(x_h.shape[1]):
            # (B, Ch) @ (Ch, Ch) -> (B, Ch)
            h[:, t] = (x_h[:, t] + linear(h[:, t - 1], self.w_h, self.b_h)).tanh()

        if self.training:

            def backward(dy: Tensor) -> Tensor:
                dy = dy.astype(self.dtype)

                if not self.return_sequence:
                    dh = zeros_like(h)
                    dh[:, -1] = dy
                else:
                    dh = dy

                self.set_dy(dh)

                dx_h = zeros_like(x_h)
                self.w_h.grad = zeros_like(self.w_h)

                for t in range(x.shape[1] - 1, -1, -1):
                    out_grad = dh[:, t]
                    if t < x_h.shape[1] - 1:
                        # hidden state gradients from next time step
                        out_grad += dx_h[:, t + 1] @ self.w_h

                    # activation gradients
                    dx_h[:, t] = (1 - h[:, t] ** 2) * out_grad

                    # hidden weight gradients
                    # (Ch, B) @ (B, Ch) -> (Ch, Ch)
                    if t > 0:
                        self.w_h.grad += dx_h[:, t].T @ h[:, t - 1]

                # hidden bias gradients
                # (B, T, Ch) -> (Ch,)
                self.b_h.grad = dx_h.sum((0, 1))

                # input gradients
                return linear_backward(dx_h, x, self.w_i, self.b_i)

            self.backward = backward

        y = h if self.return_sequence else h[:, -1]
        self.set_y(y)
        return y


class LSTM(Module):
    """Long Short-Term Memory module."""

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        return_sequence: bool = True,
        dtype: DtypeLike = "float32",
    ) -> None:
        """Long Short-Term Memory module.
        Input: (B, T, Cin)
            B ... batch, T ... time, Cin ... input channels
        Output: (B, T, Ch) if return_sequence=True else (B, Ch)
            B ... batch, T ... time, Ch ... hidden channels

        Parameters
        ----------
        in_channels : int
            Number of input features.
        h_channels : int
            Number of hidden channels.
        bias : bool, optional
            Whether to use bias values, by default True.
        return_sequence: bool, optional
            Whether to return the entire sequence or only the last hidden state.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.bias = bias
        self.return_sequence = return_sequence
        self.dtype = dtype

        k = in_channels**-0.5

        # init input weights and biases (Wii, Wif, Wig, Wio concatinated)
        w_i = uniform((4 * h_channels, in_channels), -k, k)
        self.w_i = Parameter(w_i, dtype=dtype, label="w_i")
        self.b_i = (
            Parameter(zeros((4 * h_channels,)), dtype=dtype, label="b_i")
            if bias
            else None
        )

        # init hidden weights and biases (Wii, Wif, Wig, Wio concatinated)
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
        return_sequence = self.return_sequence
        dtype = self.dtype
        return f"{name}({in_channels=}, {h_channels=}, {bias=}, {return_sequence=}, {dtype=})"

    def forward(self, x: Tensor):
        self.check_dims(x, [3])
        x = x.astype(self.dtype)
        n = self.h_channels
        n2 = 2 * n
        n3 = 3 * n

        # input
        # (B, T, Cin) @ (Cin, 4*Ch) + (4*Ch,) -> (B, T, 4*Ch)
        x_h = linear(x, self.w_i, self.b_i)

        # iterate over timesteps
        ifgo = zeros_like(x_h)
        c = zeros_like(x_h[:, :, :n])
        h = zeros_like(c)

        for t in range(x.shape[1]):
            # gates pre activation
            # (B, 4*Ch) + (B, Ch) @ (Ch, 4*Ch) + (4*Ch,) -> (B, 4*Ch)
            ifgo_preact = x_h[:, t] + linear(h[:, t - 1], self.w_h, self.b_h)

            # gates post activation i_t, f_t, g_t, o_t
            ifgo[:, t, :n2] = sigmoid(ifgo_preact[:, :n2])  # input, forget
            ifgo[:, t, n2:n3] = ifgo_preact[:, n2:n3].tanh()  # node
            ifgo[:, t, -n:] = sigmoid(ifgo_preact[:, -n:])  # output

            # cell state
            # c_t = f_t * c_t-1 + i_t * g_t
            c[:, t] = (
                ifgo[:, t, n:n2] * c[:, t - 1] + ifgo[:, t, :n] * ifgo[:, t, n2:n3]
            )

            # hidden state
            # h_t = o_t * tanh(c_t)
            h[:, t] = ifgo[:, t, -n:] * c[:, t].tanh()

        if self.training:

            def backward(dy: Tensor) -> Tensor:
                dy = dy.astype(self.dtype)

                if not self.return_sequence:
                    dh = zeros_like(h)
                    dh[:, -1] = dy
                else:
                    dh = dy

                self.set_dy(dh)

                dc = zeros_like(c)
                difgo_preact = zeros_like(ifgo)
                self.w_h.grad = zeros_like(self.w_h)

                for t in range(x.shape[1] - 1, -1, -1):
                    difgo_t = zeros_like(ifgo[:, 1])

                    out_grad = dh[:, t]
                    if t < x.shape[1] - 1:
                        # hidden state gradients from next time step
                        out_grad += difgo_preact[:, t + 1] @ self.w_h

                    # cell state gradients
                    # dc_t = dtanh(c_t) * do_t * output grads
                    dc[:, t] = (1 - c[:, t].tanh() ** 2) * ifgo[:, t, -n:] * out_grad
                    if t < x.shape[1] - 1:
                        # cell state gradients from next time step
                        # dc_t += f_t+1 * dc_t+1
                        dc[:, t] += ifgo[:, t + 1, n:n2] * dc[:, t + 1]

                    # input gate gradients
                    # di_t = g_t * dc_t
                    difgo_t[:, :n] = ifgo[:, t, n2:n3] * dc[:, t]

                    # forget gate gradients
                    # df_t = c_t-1 * dc_t
                    difgo_t[:, n:n2] = (c[:, t - 1] * dc[:, t]) if t > 0 else 0

                    # node gradients
                    # dg_t = i_t * dc_t
                    difgo_t[:, n2:n3] = ifgo[:, t, :n] * dc[:, t]

                    # output gate gradients
                    # do_t = tanh(c_t) * output grads
                    difgo_t[:, -n:] = c[:, t].tanh() * out_grad

                    # pre actiation gradients
                    # di_t, df_t = dsigmoid(i_t, f_t) * di_t, df_t
                    difgo_preact[:, t, :n2] = (
                        ifgo[:, t, :n2] * (1 - ifgo[:, t, :n2]) * difgo_t[:, :n2]
                    )

                    # dg_t = dtanh(g_t) * dg_t
                    difgo_preact[:, t, n2:n3] = (1 - ifgo[:, t, n2:n3] ** 2) * difgo_t[
                        :, n2:n3
                    ]

                    # do_t = dsigmoid(o_t) * do_t
                    difgo_preact[:, t, -n:] = (
                        ifgo[:, t, -n:] * (1 - ifgo[:, t, -n:]) * difgo_t[:, -n:]
                    )

                    # hidden weight gradients
                    # (Ch, B) @ (B, Ch) -> (Ch, Ch)
                    if t > 0:
                        self.w_h.grad += difgo_preact[:, t].T @ h[:, t - 1]

                # hidden bias gradients
                # (B, T, Ch) -> (Ch,)
                self.b_h.grad = difgo_preact.sum(axis=(0, 1))

                # input gradients
                return linear_backward(difgo_preact, x, self.w_i, self.b_i)

            self.backward = backward

        y = h if self.return_sequence else h[:, -1]
        self.set_y(y)
        return y
