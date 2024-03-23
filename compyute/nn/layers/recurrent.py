"""Recurrent layers layer"""

from compyute.functional import zeros, zeros_like
from compyute.nn.funcional import sigmoid
from compyute.nn.module import Module
from compyute.nn.parameter import Parameter
from compyute.random import uniform
from compyute.tensor import Tensor


__all__ = ["RecurrentCell"]


class RecurrentCell(Module):
    """Recurrent cell."""

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        use_bias: bool = True,
        dtype: str = "float32",
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
        use_bias : bool, optional
            Whether to use bias values, by default True.
        dtype: str, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.use_bias = use_bias
        self.dtype = dtype

        k = h_channels**-0.5

        # init input weights
        # (Cin, Ch)
        w_i = uniform((h_channels, in_channels), -k, k)
        self.w_i = Parameter(w_i, dtype=dtype, label="w_i")

        # init input biases
        # (Ch,)
        if use_bias:
            b_i = zeros((h_channels,))
            self.b_i = Parameter(b_i, dtype=dtype, label="b_i")

        # init hidden weights
        # (Ch, Ch)
        w_h = uniform((h_channels, h_channels), -k, k)
        self.w_h = Parameter(w_h, dtype=dtype, label="w_h")

        # init hidden biases
        # (Ch,)
        if use_bias:
            b_h = zeros((h_channels,))
            self.b_h = Parameter(b_h, dtype=dtype, label="b_h")

    def __repr__(self):
        name = self.__class__.__name__
        in_channels = self.in_channels
        h_channels = self.h_channels
        use_bias = self.use_bias
        dtype = self.dtype
        return f"{name}({in_channels=}, {h_channels=}, {use_bias=}, {dtype=})"

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [3])
        x = x.astype(self.dtype)

        # input projection
        # (B, T, Cin) @ (Cin, Ch) -> (B, T, Ch)
        x_h = x @ self.w_i.T
        if self.use_bias:
            # (B, T, Ch)+ (Ch,) -> (B, T, Ch)
            x_h += self.b_i

        # iterate over timesteps
        h = zeros_like(x_h, dtype=self.dtype, device=self.device)
        for t in range(x_h.shape[1]):
            # (B, Ch) @ (Ch, Ch) -> (B, Ch)
            h_t = h[:, t - 1] @ self.w_h.T
            if self.use_bias:
                h_t += self.b_h

            # activation
            h[:, t] = (x_h[:, t] + h_t).tanh()

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
                        # (B, Ch) + (B, Ch) @ (Ch, Ch) -> (B, Ch)
                        out_grad = dh[:, t] + dx_h[:, t + 1] @ self.w_h

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
                # (B, T, Ch) @ (Ch, Cin) -> (B, T, Cin)
                dx = dx_h @ self.w_i

                # input weight grads
                # (B, Ch, T) @ (B, T, Cin) -> (B, Ch, Cin)
                dw = dx_h.transpose() @ x
                # (B, Ch, Cin) -> (Ch, Cin)
                self.w_i.grad = dw.sum(axis=0)

                # input bias grads
                # (B, T, Ch) -> (Ch,)
                self.b_i.grad = dx_h.sum((0, 1))

                return dx

            self.backward = backward

        self.set_y(h)
        return h


class LSTMCell(Module):
    """Long Short-Term Memory cell."""

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        use_bias: bool = True,
        dtype: str = "float32",
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
        use_bias : bool, optional
            Whether to use bias values, by default True.
        dtype: str, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.use_bias = use_bias
        self.dtype = dtype

        k = in_channels**-0.5

        # input gate
        w_i = uniform((h_channels, in_channels), -k, k)
        self.w_i = Parameter(w_i, dtype=dtype, label="w_i")
        u_i = uniform((h_channels, h_channels), -k, k)
        self.u_i = Parameter(u_i, dtype=dtype, label="u_i")
        if use_bias:
            b_i = zeros((h_channels,))
            self.b_i = Parameter(b_i, dtype=dtype, label="b_i")

        # forget gate
        w_f = uniform((h_channels, in_channels), -k, k)
        self.w_f = Parameter(w_f, dtype=dtype, label="w_f")
        u_f = uniform((h_channels, h_channels), -k, k)
        self.u_f = Parameter(u_f, dtype=dtype, label="u_f")
        if use_bias:
            b_f = zeros((h_channels,))
            self.b_f = Parameter(b_f, dtype=dtype, label="b_f")

        # output gate
        w_o = uniform((h_channels, in_channels), -k, k)
        self.w_o = Parameter(w_o, dtype=dtype, label="w_o")
        u_o = uniform((h_channels, h_channels), -k, k)
        self.u_o = Parameter(u_o, dtype=dtype, label="u_o")
        if use_bias:
            b_o = zeros((h_channels,))
            self.b_o = Parameter(b_o, dtype=dtype, label="b_o")

        # cell
        w_c = uniform((h_channels, in_channels), -k, k)
        self.w_c = Parameter(w_c, dtype=dtype, label="w_c")
        u_c = uniform((h_channels, h_channels), -k, k)
        self.u_c = Parameter(u_c, dtype=dtype, label="u_c")
        if use_bias:
            b_c = zeros((h_channels,))
            self.b_c = Parameter(b_c, dtype=dtype, label="b_c")

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [3])
        x = x.astype(self.dtype)

        # input projections
        # (B, T, Cin) @ (Cin, Ch) -> (B, T, Ch)
        i_h = x @ self.w_i.T
        f_h = x @ self.w_f.T
        o_h = x @ self.w_i.T
        c_h = x @ self.w_i.T

        if self.use_bias:
            # (B, T, Ch)+ (Ch,) -> (B, T, Ch)
            i_h += self.b_i
            f_h += self.b_f
            o_h += self.b_o
            c_h += self.b_c

        # iterate over timesteps
        i = zeros_like(i_h, dtype=self.dtype, device=self.device)
        f = zeros_like(f_h, dtype=self.dtype, device=self.device)
        o = zeros_like(o_h, dtype=self.dtype, device=self.device)
        c = zeros_like(c_h, dtype=self.dtype, device=self.device)
        h = zeros_like(c_h, dtype=self.dtype, device=self.device)

        for t in range(x.shape[1]):
            i[:, t] = sigmoid(i_h[:, t] + h[:, t - 1] @ self.u_i.T)
            f[:, t] = sigmoid(f_h[:, t] + h[:, t - 1] @ self.u_f.T)
            o[:, t] = sigmoid(o_h[:, t] + h[:, t - 1] @ self.u_o.T)
            c_t_p = (c_h[:, t] + c[:, t - 1] @ self.u_c.T).tanh()
            c[:, t] = f[:, t] * c[:, t - 1] + i[:, t] * c_t_p
            h[:, t] = o[:, t] * c[:, t].tanh()

        self.set_y(o)
        return o
