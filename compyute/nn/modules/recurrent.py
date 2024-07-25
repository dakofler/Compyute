"""Recurrent cells module"""

from typing import Optional

from ...base_tensor import Tensor
from ...dtypes import Dtype, _DtypeLike
from ...random import uniform
from ...tensor_functions.creating import empty_like, zeros, zeros_like
from ...tensor_functions.transforming import sum as cpsum
from ...tensor_functions.transforming import tanh_
from ..functional.activations import sigmoid
from ..functional.activations import tanh as tanh
from ..functional.linear import linear
from ..parameter import Parameter
from .module import Module

__all__ = ["LSTM", "Recurrent"]


class Recurrent(Module):
    """Recurrent module."""

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

        grad_fns = []
        h = zeros((*x.shape[:2], self.h_channels), self.dtype, self.device)

        # iterate over timesteps
        for t in range(x.shape[1]):

            # input projection
            x_h, x_h_grad_fn = linear(x[:, t], self.w_i, self.b_i, self._training)

            # hidden projection
            h_h, h_h_grad_fn = linear(h[:, t - 1], self.w_h, self.b_h, self._training)

            # apply non-linearity
            h[:, t], tanh_grad_fn = tanh(x_h + h_h, self._training)

            if self._training:
                grad_fns.append((x_h_grad_fn, h_h_grad_fn, tanh_grad_fn))

        if self._training:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.as_type(self.dtype)
                dx = empty_like(x)
                dh = 0

                for t in range(x.shape[1] - 1, -1, -1):
                    # add output gradients if returning sequence or last time step
                    dh += dy[:, t] if self.return_sequence or t == x.shape[1] - 1 else 0

                    # non-linearity backward
                    dtanh = grad_fns[t][2](dh)

                    # hidden projection backward
                    dh, dw_h, db_h = grad_fns[t][1](dtanh)

                    if t > 0:
                        if self.w_h.requires_grad:
                            self.w_h.grad += dw_h
                    if self.b_h is not None and self.b_h.requires_grad:
                        self.b_h.grad += db_h

                    # input projeciton backward
                    dx[:, t], dw_i, db_i = grad_fns[t][0](dtanh)  # x_h_grad_fn

                    if self.w_i.requires_grad:
                        self.w_i.grad += dw_i
                    if self.b_i is not None and self.b_i.requires_grad:
                        self.b_i.grad += db_i

                return dx

            self._backward = _backward

        return h if self.return_sequence else h[:, -1]


class LSTM(Module):
    """Long Short-Term Memory module."""

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
        self.dtype = Dtype(dtype)

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
        x_h, x_h_grad_fn = linear(x, self.w_i, self.b_i, self._training)

        # iterate over timesteps
        gates = empty_like(x_h)
        c = zeros_like(x_h[:, :, :i1])
        h = zeros_like(c)

        for t in range(x.shape[1]):
            # gates pre activation
            # (B, 4*Ch) + (B, Ch) @ (Ch, 4*Ch) + (4*Ch,) -> (B, 4*Ch)
            h_h, _ = linear(h[:, t - 1], self.w_h, self.b_h)
            gates_preact = x_h[:, t] + h_h

            # gates post activation i_t, f_t, g_t, o_t
            gates[:, t, :i2], _ = sigmoid(gates_preact[:, :i2])  # input, forget
            gates[:, t, i2:i3] = tanh_(gates_preact[:, i2:i3])  # node
            gates[:, t, i3:], _ = sigmoid(gates_preact[:, i3:])  # output

            # cell state
            # c_t = f_t * c_t-1 + i_t * g_t
            c[:, t] = gates[:, t, i1:i2] * c[:, t - 1] + gates[:, t, :i1] * gates[:, t, i2:i3]

            # hidden state
            # h_t = o_t * tanh(c_t)
            h[:, t] = gates[:, t, i3:] * tanh_(c[:, t])

        if self._training and x_h_grad_fn is not None:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.as_type(self.dtype)

                if not self.return_sequence:
                    dh = zeros_like(h)
                    dh[:, -1] = dy
                else:
                    dh = dy

                dc = empty_like(c)
                dgates_preact = empty_like(gates)

                for t in range(x.shape[1] - 1, -1, -1):

                    # hidden state gradients
                    out_grad = dh[:, t]
                    if t < x.shape[1] - 1:
                        # hidden state gradients from next time step
                        out_grad += dgates_preact[:, t + 1] @ self.w_h

                    # cell state gradients
                    # dc_t = dtanh(c_t) * do_t * output grads
                    dc[:, t] = (1 - tanh_(c[:, t]) ** 2) * gates[:, t, i3:] * out_grad
                    if t < x.shape[1] - 1:
                        # cell state gradients from next time step
                        # dc_t += f_t+1 * dc_t+1
                        dc[:, t] += gates[:, t + 1, i1:i2] * dc[:, t + 1]

                    dgates = empty_like(gates[:, 1])

                    # input gate gradients
                    # di_t = g_t * dc_t
                    dgates[:, :i1] = gates[:, t, i2:i3] * dc[:, t]

                    # forget gate gradients
                    # df_t = c_t-1 * dc_t
                    dgates[:, i1:i2] = (c[:, t - 1] * dc[:, t]) if t > 0 else 0

                    # node gradients
                    # dg_t = i_t * dc_t
                    dgates[:, i2:i3] = gates[:, t, :i1] * dc[:, t]

                    # output gate gradients
                    # do_t = tanh(c_t) * output grads
                    dgates[:, i3:] = tanh_(c[:, t]) * out_grad

                    # pre activation input and forget gate gradients
                    # di_t, df_t = dsigmoid(i_t, f_t) * di_t, df_t
                    dgates_preact[:, t, :i2] = (
                        gates[:, t, :i2] * (1 - gates[:, t, :i2]) * dgates[:, :i2]
                    )

                    # pre activation node gradients
                    # dg_t = dtanh(g_t) * dg_t
                    dgates_preact[:, t, i2:i3] = (1 - gates[:, t, i2:i3] ** 2) * dgates[:, i2:i3]

                    # pre activation output gate gradients
                    # do_t = dsigmoid(o_t) * do_t
                    dgates_preact[:, t, i3:] = (
                        gates[:, t, i3:] * (1 - gates[:, t, i3:]) * dgates[:, i3:]
                    )

                    # hidden weight gradients
                    # (Ch, B) @ (B, Ch) -> (Ch, Ch)
                    if t > 0 and self.w_h.requires_grad:
                        self.w_h.grad += dgates_preact[:, t].T @ h[:, t - 1]

                # hidden bias gradients
                # (B, T, Ch) -> (Ch,)
                if self.b_h is not None and self.b_h.requires_grad:
                    self.b_h.grad += cpsum(dgates_preact, axis=(0, 1))

                # input projection gradients
                dx, dw_i, db_i = x_h_grad_fn(dgates_preact)

                if dw_i is not None:
                    self.w_i.grad += dw_i

                if self.b_i is not None and db_i is not None:
                    self.b_i.grad += db_i

                return dx

            self._backward = _backward

        y = h if self.return_sequence else h[:, -1]
        return y
