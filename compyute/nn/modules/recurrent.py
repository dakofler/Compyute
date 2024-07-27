"""Recurrent cells module"""

from typing import Literal, Optional

from ...base_tensor import Tensor
from ...dtypes import Dtype, _DtypeLike
from ...random import uniform
from ...tensor_functions.creating import empty, empty_like, zeros, zeros_like
from ..functional.activations import relu, sigmoid, tanh
from ..functional.linear import linear
from ..parameter import Parameter
from .module import Module

__all__ = ["GRU", "LSTM", "Recurrent"]


class Recurrent(Module):
    """Recurrent module (following the PyTorch implementation)."""

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        activation: Literal["relu", "tanh"] = "tanh",
        return_sequence: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        """Recurrent module (following the PyTorch implementation).
        Input: (B, T, Cin)
            B ... batch, T ... time, Cin ... input channels
        Output: (B, T, Ch) if return_sequence=True else (B, Ch)
            B ... batch, T ... time, Ch ... hidden channels

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        h_channels: int
            Number of hidden channels.
        bias: bool, optional
            Whether to use bias values, by default True.
        activation: Literal["relu", "tanh"], optional
            Activation function to use, by default "tanh".
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
        self.activation = relu if activation == "relu" else tanh
        self.return_sequence = return_sequence
        self.dtype = Dtype(dtype)

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

        grad_functions = []
        h = zeros((*x.shape[:2], self.h_channels), self.dtype, self.device)

        # iterate over timesteps
        for t in range(x.shape[1]):

            # input projection
            x_h, x_h_grad_fn = linear(x[:, t], self.w_i, self.b_i, self._training)

            # hidden projection
            h_h, h_h_grad_fn = linear(h[:, t - 1], self.w_h, self.b_h, self._training)

            # apply non-linearity
            h[:, t], act_grad_fn = self.activation(x_h + h_h, self._training)

            if self._training:
                grad_functions.append((x_h_grad_fn, h_h_grad_fn, act_grad_fn))

        if self._training:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.as_type(self.dtype)
                dx = empty_like(x)
                dh = 0

                # iterate backwards over timesteps
                for t in range(x.shape[1] - 1, -1, -1):
                    x_h_grad_fn, h_h_grad_fn, act_grad_fn = grad_functions[t]

                    # add output gradients if returning sequence or last time step
                    if self.return_sequence:
                        dh += dy[:, t]
                    elif t == x.shape[1] - 1:
                        dh += dy

                    # non-linearity backward
                    dact = act_grad_fn(dh)

                    # hidden projection backward
                    dh, dw_h, db_h = h_h_grad_fn(dact)

                    if t > 0 and self.w_h.requires_grad:
                        self.w_h.grad += dw_h
                    if self.b_h is not None and self.b_h.requires_grad:
                        self.b_h.grad += db_h

                    # input projeciton backward
                    dx[:, t], dw_i, db_i = x_h_grad_fn(dact)

                    if self.w_i.requires_grad:
                        self.w_i.grad += dw_i
                    if self.b_i is not None and self.b_i.requires_grad:
                        self.b_i.grad += db_i

                return dx

            self._backward = _backward

        return h if self.return_sequence else h[:, -1]


class LSTM(Module):
    """Long Short-Term Memory module (following the PyTorch implementation)."""

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        activation: Literal["relu", "tanh"] = "tanh",
        return_sequence: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        """Long Short-Term Memory module (following the PyTorch implementation).
        Input: (B, T, Cin)
            B ... batch, T ... time, Cin ... input channels
        Output: (B, T, Ch) if return_sequence=True else (B, Ch)
            B ... batch, T ... time, Ch ... hidden channels

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        h_channels: int
            Number of hidden channels.
        bias: bool, optional
            Whether to use bias values, by default True.
        activation: Literal["relu", "tanh"], optional
            Activation function to use, by default "tanh".
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
        self.activation = relu if activation == "relu" else tanh
        self.return_sequence = return_sequence
        self.dtype = Dtype(dtype)

        k = h_channels**-0.5

        # init input weights and biases
        self.w_ii = Parameter(uniform((h_channels, in_channels), -k, k, dtype), label="lstm_w_ii")
        self.b_ii = Parameter(zeros((h_channels,), dtype), label="lstm_b_ii") if bias else None
        self.w_if = Parameter(uniform((h_channels, in_channels), -k, k, dtype), label="lstm_w_if")
        self.b_if = Parameter(zeros((h_channels,), dtype), label="lstm_b_if") if bias else None
        self.w_ig = Parameter(uniform((h_channels, in_channels), -k, k, dtype), label="lstm_w_ig")
        self.b_ig = Parameter(zeros((h_channels,), dtype), label="lstm_b_ig") if bias else None
        self.w_io = Parameter(uniform((h_channels, in_channels), -k, k, dtype), label="lstm_w_io")
        self.b_io = Parameter(zeros((h_channels,), dtype), label="lstm_b_io") if bias else None

        # init hidden weights and biases
        self.w_hi = Parameter(uniform((h_channels, h_channels), -k, k, dtype), label="lstm_w_hi")
        self.b_hi = Parameter(zeros((h_channels,), dtype), label="lstm_b_hi") if bias else None
        self.w_hf = Parameter(uniform((h_channels, h_channels), -k, k, dtype), label="lstm_w_hf")
        self.b_hf = Parameter(zeros((h_channels,), dtype), label="lstm_b_hf") if bias else None
        self.w_hg = Parameter(uniform((h_channels, h_channels), -k, k, dtype), label="lstm_w_hg")
        self.b_hg = Parameter(zeros((h_channels,), dtype), label="lstm_b_hg") if bias else None
        self.w_ho = Parameter(uniform((h_channels, h_channels), -k, k, dtype), label="lstm_w_ho")
        self.b_ho = Parameter(zeros((h_channels,), dtype), label="lstm_b_ho") if bias else None

    def forward(self, x: Tensor):
        self._check_dims(x, [3])
        x = x.as_type(self.dtype)

        grad_functions = []
        i = empty((*x.shape[:2], self.h_channels), self.dtype, self.device)
        f = empty_like(i)
        g = empty_like(i)
        o = empty_like(i)
        c = zeros_like(i)
        act_c = empty_like(i)
        h = zeros_like(i)

        # iterate over timesteps
        for t in range(x.shape[1]):

            # input projection W_i * x_t + b_i
            x_t = x[:, t]
            x_i, x_i_grad_fn = linear(x_t, self.w_ii, self.b_ii, self._training)
            x_f, x_f_grad_fn = linear(x_t, self.w_if, self.b_if, self._training)
            x_g, x_g_grad_fn = linear(x_t, self.w_ig, self.b_ig, self._training)
            x_o, x_o_grad_fn = linear(x_t, self.w_io, self.b_io, self._training)

            # hidden projection W_h * h_t-1 + b_h
            h_tprev = h[:, t - 1]
            h_i, h_i_grad_fn = linear(h_tprev, self.w_hi, self.b_hi, self._training)
            h_f, h_f_grad_fn = linear(h_tprev, self.w_hf, self.b_hf, self._training)
            h_g, h_g_grad_fn = linear(h_tprev, self.w_hg, self.b_hg, self._training)
            h_o, h_o_grad_fn = linear(h_tprev, self.w_ho, self.b_ho, self._training)

            # gates
            i[:, t], i_grad_fn = sigmoid(x_i + h_i, self._training)  # input gate
            f[:, t], f_grad_fn = sigmoid(x_f + h_f, self._training)  # forget gate
            o[:, t], o_grad_fn = sigmoid(x_o + h_o, self._training)  # output gate

            # input node
            g[:, t], g_grad_fn = self.activation(x_g + h_g, self._training)

            # carry state c_t = f_t * c_t-1 + i_t * g_t
            c[:, t] = f[:, t] * c[:, t - 1] + i[:, t] * g[:, t]

            # memory state h_t = o_t * act(c_t)
            act_c[:, t], actc_grad_fn = self.activation(c[:, t], self._training)
            h[:, t] = o[:, t] * act_c[:, t]

            # remember gradient functions
            if self._training:
                grad_functions_t = (
                    x_i_grad_fn,
                    x_f_grad_fn,
                    x_g_grad_fn,
                    x_o_grad_fn,
                    h_i_grad_fn,
                    h_f_grad_fn,
                    h_g_grad_fn,
                    h_o_grad_fn,
                    i_grad_fn,
                    f_grad_fn,
                    g_grad_fn,
                    o_grad_fn,
                    actc_grad_fn,
                )
                grad_functions.append(grad_functions_t)

        if self._training:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.as_type(self.dtype)
                dx = empty_like(x)
                dc = zeros_like(c)
                dh = 0

                # iterate backwards over timesteps
                for t in range(x.shape[1] - 1, -1, -1):

                    # load gradient functions
                    (
                        x_i_grad_fn,
                        x_f_grad_fn,
                        x_g_grad_fn,
                        x_o_grad_fn,
                        h_i_grad_fn,
                        h_f_grad_fn,
                        h_g_grad_fn,
                        h_o_grad_fn,
                        i_grad_fn,
                        f_grad_fn,
                        g_grad_fn,
                        o_grad_fn,
                        actc_grad_fn,
                    ) = grad_functions[t]

                    # add output gradients if returning sequence or last time step
                    if self.return_sequence:
                        dh += dy[:, t]
                    elif t == x.shape[1] - 1:
                        dh += dy

                    # memory state gradients
                    do = act_c[:, t] * dh
                    dc[:, t] += actc_grad_fn(dh) * o[:, t]

                    # carry state gradients
                    df = c[:, t - 1] * dc[:, t] if t > 0 else 0
                    if t > 0:
                        dc[:, t - 1] += f[:, t] * dc[:, t]
                    di = g[:, t] * dc[:, t]
                    dg = i[:, t] * dc[:, t]

                    # gate gradients
                    di_preact = i_grad_fn(di)
                    df_preact = f_grad_fn(df)
                    do_preact = o_grad_fn(do)

                    # input node gradients
                    dg_preact = g_grad_fn(dg)

                    # hidden projection gradients
                    dh_i, dw_hi, db_hi = h_i_grad_fn(di_preact)

                    if t > 0 and self.w_hi.requires_grad:
                        self.w_hi.grad += dw_hi
                    if self.b_hi is not None and self.b_hi.requires_grad:
                        self.b_hi.grad += db_hi

                    dh_f, dw_hf, db_hf = h_f_grad_fn(df_preact)
                    if t > 0 and self.w_hf.requires_grad:
                        self.w_hf.grad += dw_hf
                    if self.b_hf is not None and self.b_hf.requires_grad:
                        self.b_hf.grad += db_hf

                    dh_g, dw_hg, db_hg = h_g_grad_fn(dg_preact)
                    if t > 0 and self.w_hg.requires_grad:
                        self.w_hg.grad += dw_hg
                    if self.b_hg is not None and self.b_hg.requires_grad:
                        self.b_hg.grad += db_hg

                    dh_o, dw_ho, db_ho = h_o_grad_fn(do_preact)
                    if t > 0 and self.w_ho.requires_grad:
                        self.w_ho.grad += dw_ho
                    if self.b_ho is not None and self.b_ho.requires_grad:
                        self.b_ho.grad += db_ho

                    dh = dh_i + dh_f + dh_g + dh_o

                    # input projection gradients
                    dx_i, dw_ii, db_ii = x_i_grad_fn(di_preact)

                    if self.w_ii.requires_grad:
                        self.w_ii.grad += dw_ii
                    if self.b_ii is not None and self.b_ii.requires_grad:
                        self.b_ii.grad += db_ii

                    dx_f, dw_if, db_if = x_f_grad_fn(df_preact)
                    if self.w_if.requires_grad:
                        self.w_if.grad += dw_if
                    if self.b_if is not None and self.b_if.requires_grad:
                        self.b_if.grad += db_if

                    dx_g, dw_ig, db_ig = x_g_grad_fn(dg_preact)
                    if self.w_ig.requires_grad:
                        self.w_ig.grad += dw_ig
                    if self.b_ig is not None and self.b_ig.requires_grad:
                        self.b_ig.grad += db_ig

                    dx_o, dw_io, db_io = x_o_grad_fn(do_preact)
                    if self.w_io.requires_grad:
                        self.w_io.grad += dw_io
                    if self.b_io is not None and self.b_io.requires_grad:
                        self.b_io.grad += db_io

                    dx[:, t] = dx_i + dx_f + dx_g + dx_o
                return dx

            self._backward = _backward

        return h if self.return_sequence else h[:, -1]


class GRU(Module):
    """Gated Recurrent Unit module (following the PyTorch implementation)."""

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        activation: Literal["relu", "tanh"] = "tanh",
        return_sequence: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        """Gated Recurrent Unit module (following the PyTorch implementation).
        Input: (B, T, Cin)
            B ... batch, T ... time, Cin ... input channels
        Output: (B, T, Ch) if return_sequence=True else (B, Ch)
            B ... batch, T ... time, Ch ... hidden channels

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        h_channels: int
            Number of hidden channels.
        bias: bool, optional
            Whether to use bias values, by default True.
        activation: Literal["relu", "tanh"], optional
            Activation function to use, by default "tanh".
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
        self.activation = relu if activation == "relu" else tanh
        self.return_sequence = return_sequence
        self.dtype = Dtype(dtype)

        k = h_channels**-0.5

        # init input weights and biases
        self.w_ir = Parameter(uniform((h_channels, in_channels), -k, k, dtype), label="lstm_w_ir")
        self.b_ir = Parameter(zeros((h_channels,), dtype), label="lstm_b_ir") if bias else None
        self.w_iz = Parameter(uniform((h_channels, in_channels), -k, k, dtype), label="lstm_w_iz")
        self.b_iz = Parameter(zeros((h_channels,), dtype), label="lstm_b_iz") if bias else None
        self.w_in = Parameter(uniform((h_channels, in_channels), -k, k, dtype), label="lstm_w_in")
        self.b_in = Parameter(zeros((h_channels,), dtype), label="lstm_b_in") if bias else None

        # init hidden weights and biases
        self.w_hr = Parameter(uniform((h_channels, h_channels), -k, k, dtype), label="lstm_w_hr")
        self.b_hr = Parameter(zeros((h_channels,), dtype), label="lstm_b_hr") if bias else None
        self.w_hz = Parameter(uniform((h_channels, h_channels), -k, k, dtype), label="lstm_w_hz")
        self.b_hz = Parameter(zeros((h_channels,), dtype), label="lstm_b_hz") if bias else None
        self.w_hn = Parameter(uniform((h_channels, h_channels), -k, k, dtype), label="lstm_w_hn")
        self.b_hn = Parameter(zeros((h_channels,), dtype), label="lstm_b_hn") if bias else None

    def forward(self, x: Tensor):
        self._check_dims(x, [3])
        x = x.as_type(self.dtype)

        grad_functions = []
        r = empty((*x.shape[:2], self.h_channels), self.dtype, self.device)
        z = empty_like(r)
        n = empty_like(r)
        h_n = empty_like(r)
        h = zeros_like(r)

        # iterate over timesteps
        for t in range(x.shape[1]):

            # input projection W_i * x_t + b_i
            x_t = x[:, t]
            x_r, x_r_grad_fn = linear(x_t, self.w_ir, self.b_ir, self._training)
            x_z, x_z_grad_fn = linear(x_t, self.w_iz, self.b_iz, self._training)
            x_n, x_n_grad_fn = linear(x_t, self.w_in, self.b_in, self._training)

            # hidden projection W_h * h_t-1 + b_h
            h_tprev = h[:, t - 1]
            h_r, h_r_grad_fn = linear(h_tprev, self.w_hr, self.b_hr, self._training)
            h_z, h_z_grad_fn = linear(h_tprev, self.w_hz, self.b_hz, self._training)
            h_n[:, t], h_n_grad_fn = linear(h_tprev, self.w_hn, self.b_hn, self._training)

            # gates
            r[:, t], r_grad_fn = sigmoid(x_r + h_r, self._training)  # reset gate
            z[:, t], z_grad_fn = sigmoid(x_z + h_z, self._training)  # update gate

            # candidate hidden state n_t = act(x_n + r_t * h_t-1)
            n[:, t], n_grad_fn = self.activation(x_n + r[:, t] * h_n[:, t], self._training)

            # hidden state h_t = (1 - z_t) * n_t + z_t * h_t-1
            h[:, t] = (1 - z[:, t]) * n[:, t] + z[:, t] * h[:, t - 1]

            # remember gradient functions
            if self._training:
                grad_functions_t = (
                    x_r_grad_fn,
                    x_z_grad_fn,
                    x_n_grad_fn,
                    h_r_grad_fn,
                    h_z_grad_fn,
                    h_n_grad_fn,
                    r_grad_fn,
                    z_grad_fn,
                    n_grad_fn,
                )
                grad_functions.append(grad_functions_t)

        if self._training:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.as_type(self.dtype)
                dx = empty_like(x)
                dh = 0

                # iterate backwards over timesteps
                for t in range(x.shape[1] - 1, -1, -1):

                    # load gradient functions
                    (
                        x_r_grad_fn,
                        x_z_grad_fn,
                        x_n_grad_fn,
                        h_r_grad_fn,
                        h_z_grad_fn,
                        h_n_grad_fn,
                        r_grad_fn,
                        z_grad_fn,
                        n_grad_fn,
                    ) = grad_functions[t]

                    # add output gradients if returning sequence or last time step
                    if self.return_sequence:
                        dh += dy[:, t]
                    elif t == x.shape[1] - 1:
                        dh += dy

                    # hidden state gradients
                    dz = ((h[:, t - 1] if t > 0 else 0) - n[:, t]) * dh
                    dn = (1 - z[:, t]) * dh
                    dh = z[:, t] * dh

                    # candidate hidden state gradients
                    dn_preact = n_grad_fn(dn)
                    dr = h_n[:, t] * dn_preact

                    # gate gradients
                    dr_preact = r_grad_fn(dr)
                    dz_preact = z_grad_fn(dz)

                    # hidden projection gradients
                    dh_r, dw_hr, db_hr = h_r_grad_fn(dr_preact)

                    if t > 0 and self.w_hr.requires_grad:
                        self.w_hr.grad += dw_hr
                    if self.b_hr is not None and self.b_hr.requires_grad:
                        self.b_hr.grad += db_hr

                    dh_z, dw_hz, db_hz = h_z_grad_fn(dz_preact)
                    if t > 0 and self.w_hz.requires_grad:
                        self.w_hz.grad += dw_hz
                    if self.b_hz is not None and self.b_hz.requires_grad:
                        self.b_hz.grad += db_hz

                    dh_n, dw_hn, db_hn = h_n_grad_fn(r[:, t] * dn_preact)
                    if t > 0 and self.w_hn.requires_grad:
                        self.w_hn.grad += dw_hn
                    if self.b_hn is not None and self.b_hn.requires_grad:
                        self.b_hn.grad += db_hn

                    dh += dh_r + dh_z + dh_n

                    # input projection gradients
                    dx_r, dw_ir, db_ir = x_r_grad_fn(dr_preact)
                    if self.w_ir.requires_grad:
                        self.w_ir.grad += dw_ir
                    if self.b_ir is not None and self.b_ir.requires_grad:
                        self.b_ir.grad += db_ir

                    dx_z, dw_iz, db_iz = x_z_grad_fn(dz_preact)
                    if self.w_iz.requires_grad:
                        self.w_iz.grad += dw_iz
                    if self.b_iz is not None and self.b_iz.requires_grad:
                        self.b_iz.grad += db_iz

                    dx_n, dw_in, db_in = x_n_grad_fn(dn_preact)
                    if self.w_in.requires_grad:
                        self.w_in.grad += dw_in
                    if self.b_in is not None and self.b_in.requires_grad:
                        self.b_in.grad += db_in

                    dx[:, t] = dx_r + dx_z + dx_n
                return dx

            self._backward = _backward

        return h if self.return_sequence else h[:, -1]
