"""Recurrent cells module"""

from typing import Optional

from ...base_tensor import Tensor
from ...dtypes import Dtype, _DtypeLike
from ...random import uniform
from ...tensor_functions.creating import empty_like, zeros, zeros_like
from ...tensor_functions.transforming import sum as _sum
from ...tensor_functions.transforming import tanh
from ..functional.activations import sigmoid
from ..functional.linear import linear
from ..parameter import Parameter
from .module import Module

__all__ = ["LSTM", "Recurrent"]


class Recurrent(Module):
    """Recurrent module."""

    __slots__ = (
        "in_channels",
        "h_channels",
        "bias",
        "return_sequence",
        "dtype",
        "w_i",
        "b_i",
        "w_h",
        "b_h",
    )

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        return_sequence: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        """Recurrent module.
        Input: (B, T, Cin)
            B ... batch, T ... time, Cin ... input channels
        Output: (B, T, Ch) if return_sequence=True else (B, Ch)
            B ... batch, T ... time, Ch ... hidden channels

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        h_channels : int
            Number of hidden channels.
        bias : bool, optional
            Whether to use bias values, by default True.
        return_sequence: bool, optional
            Whether to return the entire sequence or only the last hidden state.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default Dtype.FLOAT32.
        label: str, optional
            Module label.
        training: bool, optional
            Whether the module should be in training mode, by default False.
        """
        super().__init__(label, training)
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.bias = bias
        self.return_sequence = return_sequence
        self.dtype = dtype

        k = h_channels**-0.5

        # init input weights and biases
        self.w_i = Parameter(uniform((h_channels, in_channels), -k, k, dtype), label="rec_w_i")
        self.b_i = Parameter(zeros((h_channels,), dtype), label="rec_b_i") if bias else None

        # init hidden weights and biases
        self.w_h = Parameter(uniform((h_channels, h_channels), -k, k, dtype), label="rec_w_h")
        self.b_h = Parameter(zeros((h_channels,), dtype), label="rec_b_h") if bias else None

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [3])
        x = x.as_type(self.dtype)

        # input projection
        # (B, T, Cin) @ (Cin, Ch) -> (B, T, Ch)
        x_h, x_h_backward = linear(x, self.w_i, self.b_i, self._training)

        # iterate over timesteps
        h = zeros_like(x_h)
        for t in range(x_h.shape[1]):
            # hidden state
            # (B, Ch) @ (Ch, Ch) -> (B, Ch)
            h_h, _ = linear(h[:, t - 1], self.w_h, self.b_h)
            h[:, t] = tanh(x_h[:, t] + h_h)

        if self._training:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.as_type(self.dtype)

                if not self.return_sequence:
                    dh = zeros_like(h)
                    dh[:, -1] = dy
                else:
                    dh = dy

                dx_h = empty_like(x_h)

                for t in range(x.shape[1] - 1, -1, -1):
                    # hidden state gradients
                    out_grad = dh[:, t]
                    if t < x_h.shape[1] - 1:
                        # hidden state gradients from next time step
                        out_grad += dx_h[:, t + 1] @ self.w_h

                    # activation gradients
                    dx_h[:, t] = (1 - h[:, t] ** 2) * out_grad

                    # hidden weight gradients
                    # (Ch, B) @ (B, Ch) -> (Ch, Ch)
                    if t > 0 and self.w_h.requires_grad:
                        self.w_h.grad += dx_h[:, t].T @ h[:, t - 1]

                # hidden bias gradients
                # (B, T, Ch) -> (Ch,)
                if self.b_h is not None and self.b_h.requires_grad:
                    self.b_h.grad += _sum(dx_h, axis=(0, 1))

                # input projection gradients
                dx, dw_i, db_i = x_h_backward(dx_h)

                if dw_i is not None:
                    self.w_i.grad += dw_i

                if db_i is not None:
                    self.b_i.grad += db_i

                return dx

            self._backward = _backward

        y = h if self.return_sequence else h[:, -1]
        return y


class LSTM(Module):
    """Long Short-Term Memory module."""

    __slots__ = (
        "in_channels",
        "h_channels",
        "bias",
        "return_sequence",
        "dtype",
        "w_i",
        "b_i",
        "w_h",
        "b_h",
    )

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        return_sequence: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        """Long Short-Term Memory module.
        Input: (B, T, Cin)
            B ... batch, T ... time, Cin ... input channels
        Output: (B, T, Ch) if return_sequence=True else (B, Ch)
            B ... batch, T ... time, Ch ... hidden channels

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        h_channels : int
            Number of hidden channels.
        bias : bool, optional
            Whether to use bias values, by default True.
        return_sequence: bool, optional
            Whether to return the entire sequence or only the last hidden state.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default Dtype.FLOAT32.
        label: str, optional
            Module label.
        training: bool, optional
            Whether the module should be in training mode, by default False.
        """
        super().__init__(label, training)
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.bias = bias
        self.return_sequence = return_sequence
        self.dtype = dtype

        k = in_channels**-0.5

        # init input weights and biases
        self.w_i = Parameter(uniform((4 * h_channels, in_channels), -k, k, dtype), label="lstm_w_i")
        self.b_i = Parameter(zeros((4 * h_channels,), dtype), label="lstm_b_i") if bias else None

        # init hidden weights and biases
        self.w_h = Parameter(uniform((4 * h_channels, h_channels), -k, k, dtype), label="lstm_w_h")
        self.b_h = Parameter(zeros((4 * h_channels,), dtype), label="lstm_b_h") if bias else None

    def forward(self, x: Tensor):
        self._check_dims(x, [3])
        x = x.as_type(self.dtype)

        # indices used to access the concatinated matrices
        i1 = self.h_channels
        i2 = 2 * i1
        i3 = 3 * i1

        # input projection
        # (B, T, Cin) @ (Cin, 4*Ch) + (4*Ch,) -> (B, T, 4*Ch)
        x_h, x_h_backward = linear(x, self.w_i, self.b_i, self._training)

        # iterate over timesteps
        ifgo = empty_like(x_h)
        c = zeros_like(x_h[:, :, :i1])
        h = zeros_like(c)

        for t in range(x.shape[1]):
            # gates pre activation
            # (B, 4*Ch) + (B, Ch) @ (Ch, 4*Ch) + (4*Ch,) -> (B, 4*Ch)
            h_h, _ = linear(h[:, t - 1], self.w_h, self.b_h)
            ifgo_preact = x_h[:, t] + h_h

            # gates post activation i_t, f_t, g_t, o_t
            ifgo[:, t, :i2] = sigmoid(ifgo_preact[:, :i2])[0]  # input, forget
            ifgo[:, t, i2:i3] = tanh(ifgo_preact[:, i2:i3])  # node
            ifgo[:, t, i3:] = sigmoid(ifgo_preact[:, i3:])[0]  # output

            # cell state
            # c_t = f_t * c_t-1 + i_t * g_t
            c[:, t] = ifgo[:, t, i1:i2] * c[:, t - 1] + ifgo[:, t, :i1] * ifgo[:, t, i2:i3]

            # hidden state
            # h_t = o_t * tanh(c_t)
            h[:, t] = ifgo[:, t, i3:] * tanh(c[:, t])

        if self._training:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.as_type(self.dtype)

                if not self.return_sequence:
                    dh = zeros_like(h)
                    dh[:, -1] = dy
                else:
                    dh = dy

                dc = empty_like(c)
                difgo_preact = empty_like(ifgo)

                for t in range(x.shape[1] - 1, -1, -1):

                    # hidden state gradients
                    out_grad = dh[:, t]
                    if t < x.shape[1] - 1:
                        # hidden state gradients from next time step
                        out_grad += difgo_preact[:, t + 1] @ self.w_h

                    # cell state gradients
                    # dc_t = dtanh(c_t) * do_t * output grads
                    dc[:, t] = (1 - tanh(c[:, t]) ** 2) * ifgo[:, t, i3:] * out_grad
                    if t < x.shape[1] - 1:
                        # cell state gradients from next time step
                        # dc_t += f_t+1 * dc_t+1
                        dc[:, t] += ifgo[:, t + 1, i1:i2] * dc[:, t + 1]

                    difgo_t = empty_like(ifgo[:, 1])

                    # input gate gradients
                    # di_t = g_t * dc_t
                    difgo_t[:, :i1] = ifgo[:, t, i2:i3] * dc[:, t]

                    # forget gate gradients
                    # df_t = c_t-1 * dc_t
                    difgo_t[:, i1:i2] = (c[:, t - 1] * dc[:, t]) if t > 0 else 0

                    # node gradients
                    # dg_t = i_t * dc_t
                    difgo_t[:, i2:i3] = ifgo[:, t, :i1] * dc[:, t]

                    # output gate gradients
                    # do_t = tanh(c_t) * output grads
                    difgo_t[:, i3:] = tanh(c[:, t]) * out_grad

                    # pre activation input and forget gate gradients
                    # di_t, df_t = dsigmoid(i_t, f_t) * di_t, df_t
                    difgo_preact[:, t, :i2] = (
                        ifgo[:, t, :i2] * (1 - ifgo[:, t, :i2]) * difgo_t[:, :i2]
                    )

                    # pre activation node gradients
                    # dg_t = dtanh(g_t) * dg_t
                    difgo_preact[:, t, i2:i3] = (1 - ifgo[:, t, i2:i3] ** 2) * difgo_t[:, i2:i3]

                    # pre activation output gate gradients
                    # do_t = dsigmoid(o_t) * do_t
                    difgo_preact[:, t, i3:] = (
                        ifgo[:, t, i3:] * (1 - ifgo[:, t, i3:]) * difgo_t[:, i3:]
                    )

                    # hidden weight gradients
                    # (Ch, B) @ (B, Ch) -> (Ch, Ch)
                    if t > 0 and self.w_h.requires_grad:
                        self.w_h.grad += difgo_preact[:, t].T @ h[:, t - 1]

                # hidden bias gradients
                # (B, T, Ch) -> (Ch,)
                if self.b_h is not None and self.b_h.requires_grad:
                    self.b_h.grad += _sum(difgo_preact, axis=(0, 1))

                # input projection gradients
                dx, dw_i, db_i = x_h_backward(difgo_preact)

                if dw_i is not None:
                    self.w_i.grad += dw_i

                if db_i is not None:
                    self.b_i.grad += db_i

                return dx

            self._backward = _backward

        y = h if self.return_sequence else h[:, -1]
        return y
