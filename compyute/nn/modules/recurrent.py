"""Neural network recurrent modules."""

import math
from typing import Literal, Optional

from ...random.random import uniform
from ...tensor_ops.creating import empty, empty_like, zeros, zeros_like
from ...tensors import Tensor
from ...typing import DType
from ..functional.activations import relu, sigmoid, tanh
from ..functional.linear import linear
from ..parameter import Parameter, update_parameter_grad
from .module import Module, validate_input_axes

__all__ = ["GRU", "LSTM", "Recurrent"]


class Recurrent(Module):
    r"""Elman Recurrent module.
    For each element in the sequence the hidden state is computed as follows:

    .. math::
        h_t = tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T + b_{hh})

    .. note::
        This implementation follows the one by PyTorch which differs slightly from the original paper.

    Shapes:
        - Input :math:`(B, S, C_{in})`
        - Output :math:`(B, S, C_{h})` if ``return_sequence=True`` else :math:`(B, C_{h})`
    where
        - :math:`B` ... batch axis
        - :math:`S` ... sequence
        - :math:`C_{in}` ... input channels
        - :math:`C_{h}` ... hidden channels

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    h_channels : int
        Number of hidden channels.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    activation : Literal["relu", "tanh"], optional
        Activation function to use. Defaults to ``tanh``.
    return_sequence : bool, optional
        Whether to return the entire sequence or only the last hidden state.
    dtype : DType, optional
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{h}}}`. Biases are initialized as zeros.
    """

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        activation: Literal["relu", "tanh"] = "tanh",
        return_sequence: bool = True,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.bias = bias
        self.act = relu if activation == "relu" else tanh
        self.return_sequence = return_sequence

        k = 1 / math.sqrt(h_channels)

        # init input weights and biases
        self.w_i = Parameter(uniform((h_channels, in_channels), -k, k, dtype=dtype))
        self.b_i = Parameter(zeros((h_channels,), dtype=dtype)) if bias else None

        # init hidden weights and biases
        self.w_h = Parameter(uniform((h_channels, h_channels), -k, k, dtype=dtype))
        self.b_h = Parameter(zeros((h_channels,), dtype=dtype)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [3])

        grad_functions = []
        h = zeros((*x.shape[:2], self.h_channels), x.device, x.dtype)

        # iterate over timesteps
        for t in range(x.shape[1]):

            # input projection
            x_t = x[:, t]
            x_h, x_h_grad_fn = linear(x_t, self.w_i, self.b_i, self._is_training)

            # hidden projection
            h_prev = h[:, t - 1]
            h_h, h_h_grad_fn = linear(h_prev, self.w_h, self.b_h, self._is_training)

            # apply non-linearity
            h[:, t], act_grad_fn = self.act(x_h + h_h, self._is_training)

            if self._is_training:
                grad_functions.append((x_h_grad_fn, h_h_grad_fn, act_grad_fn))

        if self._is_training:

            def _backward(dy: Tensor) -> Tensor:
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
                    if t > 0:
                        update_parameter_grad(self.w_h, dw_h)
                    update_parameter_grad(self.b_h, db_h)

                    # input projeciton backward
                    dx[:, t], dw_i, db_i = x_h_grad_fn(dact)
                    update_parameter_grad(self.w_i, dw_i)
                    update_parameter_grad(self.b_i, db_i)

                return dx

            self._backward = _backward

        return h if self.return_sequence else h[:, -1]


class LSTM(Module):
    r"""Long Short-Term Memory module.
    For each element in the sequence the hidden state is computed as follows:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(x_tW_{ii}^T + b_{ii} + h_{t-1}W_{hi}^T + b_{hi}) \\
            f_t = \sigma(x_tW_{if}^T + b_{if} + h_{t-1}W_{hf}^T + b_{hf}) \\
            g_t = act(x_tW_{ig}^T + b_{ig} + h_{t-1}W_{hg}^T + b_{hg}) \\
            o_t = \sigma(x_tW_{io}^T + b_{io} + h_{t-1}W_{ho}^T + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot act(c_t) \\
        \end{array}

    where :math:`\odot` is the Hadamard product.

    .. note::
        This implementation follows the one by PyTorch which differs slightly from the original paper.

    Shapes:
        - Input :math:`(B, S, C_{in})`
        - Output :math:`(B, S, C_{h})` if ``return_sequence=True`` else :math:`(B, C_{h})`
    where
        - :math:`B` ... batch axis
        - :math:`S` ... sequence
        - :math:`C_{in}` ... input channels
        - :math:`C_{h}` ... hidden channels

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    h_channels : int
        Number of hidden channels.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    activation : Literal["relu", "tanh"], optional
        Activation function to use. Defaults to ``tanh``.
    return_sequence : bool, optional
        Whether to return the entire sequence or only the last hidden state.
    dtype : DType, optional
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{h}}}`. Biases are initialized as zeros.
    """

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        activation: Literal["relu", "tanh"] = "tanh",
        return_sequence: bool = True,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.bias = bias
        self.act = relu if activation == "relu" else tanh
        self.return_sequence = return_sequence

        k = 1 / math.sqrt(h_channels)

        # init input weights and biases
        self.w_ii = Parameter(uniform((h_channels, in_channels), -k, k, dtype=dtype))
        self.b_ii = Parameter(zeros((h_channels,), dtype=dtype)) if bias else None
        self.w_if = Parameter(uniform((h_channels, in_channels), -k, k, dtype=dtype))
        self.b_if = Parameter(zeros((h_channels,), dtype=dtype)) if bias else None
        self.w_ig = Parameter(uniform((h_channels, in_channels), -k, k, dtype=dtype))
        self.b_ig = Parameter(zeros((h_channels,), dtype=dtype)) if bias else None
        self.w_io = Parameter(uniform((h_channels, in_channels), -k, k, dtype=dtype))
        self.b_io = Parameter(zeros((h_channels,), dtype=dtype)) if bias else None

        # init hidden weights and biases
        self.w_hi = Parameter(uniform((h_channels, h_channels), -k, k, dtype=dtype))
        self.b_hi = Parameter(zeros((h_channels,), dtype=dtype)) if bias else None
        self.w_hf = Parameter(uniform((h_channels, h_channels), -k, k, dtype=dtype))
        self.b_hf = Parameter(zeros((h_channels,), dtype=dtype)) if bias else None
        self.w_hg = Parameter(uniform((h_channels, h_channels), -k, k, dtype=dtype))
        self.b_hg = Parameter(zeros((h_channels,), dtype=dtype)) if bias else None
        self.w_ho = Parameter(uniform((h_channels, h_channels), -k, k, dtype=dtype))
        self.b_ho = Parameter(zeros((h_channels,), dtype=dtype)) if bias else None

    def forward(self, x: Tensor):
        validate_input_axes(self, x, [3])

        grad_functions = []
        i = empty((*x.shape[:2], self.h_channels), x.device, x.dtype)
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
            x_i, x_i_grad_fn = linear(x_t, self.w_ii, self.b_ii, self._is_training)
            x_f, x_f_grad_fn = linear(x_t, self.w_if, self.b_if, self._is_training)
            x_g, x_g_grad_fn = linear(x_t, self.w_ig, self.b_ig, self._is_training)
            x_o, x_o_grad_fn = linear(x_t, self.w_io, self.b_io, self._is_training)

            # hidden projection W_h * h_t-1 + b_h
            h_prev = h[:, t - 1]
            h_i, h_i_grad_fn = linear(h_prev, self.w_hi, self.b_hi, self._is_training)
            h_f, h_f_grad_fn = linear(h_prev, self.w_hf, self.b_hf, self._is_training)
            h_g, h_g_grad_fn = linear(h_prev, self.w_hg, self.b_hg, self._is_training)
            h_o, h_o_grad_fn = linear(h_prev, self.w_ho, self.b_ho, self._is_training)

            # gates
            i[:, t], i_grad_fn = sigmoid(x_i + h_i, self._is_training)  # input gate
            f[:, t], f_grad_fn = sigmoid(x_f + h_f, self._is_training)  # forget gate
            o[:, t], o_grad_fn = sigmoid(x_o + h_o, self._is_training)  # output gate

            # input node
            g[:, t], g_grad_fn = self.act(x_g + h_g, self._is_training)

            # carry state c_t = f_t * c_t-1 + i_t * g_t
            c[:, t] = f[:, t] * c[:, t - 1] + i[:, t] * g[:, t]

            # memory state h_t = o_t * act(c_t)
            act_c[:, t], actc_grad_fn = self.act(c[:, t], self._is_training)
            h[:, t] = o[:, t] * act_c[:, t]

            # remember gradient functions
            if self._is_training:
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

        if self._is_training:

            def _backward(dy: Tensor) -> Tensor:
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
                    if t > 0:
                        update_parameter_grad(self.w_hi, dw_hi)
                    update_parameter_grad(self.b_hi, db_hi)

                    dh_f, dw_hf, db_hf = h_f_grad_fn(df_preact)
                    if t > 0:
                        update_parameter_grad(self.w_hf, dw_hf)
                    update_parameter_grad(self.b_hf, db_hf)

                    dh_g, dw_hg, db_hg = h_g_grad_fn(dg_preact)
                    if t > 0:
                        update_parameter_grad(self.w_hg, dw_hg)
                    update_parameter_grad(self.b_hg, db_hg)

                    dh_o, dw_ho, db_ho = h_o_grad_fn(do_preact)
                    if t > 0:
                        update_parameter_grad(self.w_ho, dw_ho)
                    update_parameter_grad(self.b_ho, db_ho)

                    dh = dh_i + dh_f + dh_g + dh_o

                    # input projection gradients
                    dx_i, dw_ii, db_ii = x_i_grad_fn(di_preact)
                    update_parameter_grad(self.w_ii, dw_ii)
                    update_parameter_grad(self.b_ii, db_ii)

                    dx_f, dw_if, db_if = x_f_grad_fn(df_preact)
                    update_parameter_grad(self.w_if, dw_if)
                    update_parameter_grad(self.b_if, db_if)

                    dx_g, dw_ig, db_ig = x_g_grad_fn(dg_preact)
                    update_parameter_grad(self.w_ig, dw_ig)
                    update_parameter_grad(self.b_ig, db_ig)

                    dx_o, dw_io, db_io = x_o_grad_fn(do_preact)
                    update_parameter_grad(self.w_io, dw_io)
                    update_parameter_grad(self.b_io, db_io)

                    dx[:, t] = dx_i + dx_f + dx_g + dx_o
                return dx

            self._backward = _backward

        return h if self.return_sequence else h[:, -1]


class GRU(Module):
    r"""Gated Recurrent Unit module.
    For each element in the sequence the hidden state is computed as follows:

    .. math::
        \begin{array}{ll} \\
            r_t = \sigma(x_tW_{ir}^T + b_{ir} + h_{t-1}W_{hr}^T + b_{hr}) \\
            z_t = \sigma(x_tW_{iz}^T + b_{iz} + h_{t-1}W_{hz}^T + b_{hz}) \\
            n_t = act(x_tW_{in}^T + b_{in} + r_t \odot (h_{t-1}W_{hn}^T + b_{hn})) \\
            h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}
        \end{array}

    where :math:`\odot` is the Hadamard product.
    
    .. note::
        This implementation follows the one by PyTorch which differs slightly from the original paper.

    Shapes:
        - Input :math:`(B, S, C_{in})`
        - Output :math:`(B, S, C_{h})` if ``return_sequence=True`` else :math:`(B, C_{h})`
    where
        - :math:`B` ... batch axis
        - :math:`S` ... sequence
        - :math:`C_{in}` ... input channels
        - :math:`C_{h}` ... hidden channels

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    h_channels : int
        Number of hidden channels.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    activation : Literal["relu", "tanh"], optional
        Activation function to use. Defaults to ``tanh``.
    return_sequence : bool, optional
        Whether to return the entire sequence or only the last hidden state.
    dtype : DType, optional
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{h}}}`. Biases are initialized as zeros.
    """

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        activation: Literal["relu", "tanh"] = "tanh",
        return_sequence: bool = True,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.bias = bias
        self.act = relu if activation == "relu" else tanh
        self.return_sequence = return_sequence

        k = 1 / math.sqrt(h_channels)

        # init input weights and biases
        self.w_ir = Parameter(uniform((h_channels, in_channels), -k, k, dtype=dtype))
        self.b_ir = Parameter(zeros((h_channels,), dtype=dtype)) if bias else None
        self.w_iz = Parameter(uniform((h_channels, in_channels), -k, k, dtype=dtype))
        self.b_iz = Parameter(zeros((h_channels,), dtype=dtype)) if bias else None
        self.w_in = Parameter(uniform((h_channels, in_channels), -k, k, dtype=dtype))
        self.b_in = Parameter(zeros((h_channels,), dtype=dtype)) if bias else None

        # init hidden weights and biases
        self.w_hr = Parameter(uniform((h_channels, h_channels), -k, k, dtype=dtype))
        self.b_hr = Parameter(zeros((h_channels,), dtype=dtype)) if bias else None
        self.w_hz = Parameter(uniform((h_channels, h_channels), -k, k, dtype=dtype))
        self.b_hz = Parameter(zeros((h_channels,), dtype=dtype)) if bias else None
        self.w_hn = Parameter(uniform((h_channels, h_channels), -k, k, dtype=dtype))
        self.b_hn = Parameter(zeros((h_channels,), dtype=dtype)) if bias else None

    def forward(self, x: Tensor):
        validate_input_axes(self, x, [3])

        grad_functions = []
        r = empty((*x.shape[:2], self.h_channels), x.device, x.dtype)
        z = empty_like(r)
        n = empty_like(r)
        h_n = empty_like(r)
        h = zeros_like(r)

        # iterate over timesteps
        for t in range(x.shape[1]):

            # input projection W_i * x_t + b_i
            x_t = x[:, t]
            x_r, x_r_grad_fn = linear(x_t, self.w_ir, self.b_ir, self._is_training)
            x_z, x_z_grad_fn = linear(x_t, self.w_iz, self.b_iz, self._is_training)
            x_n, x_n_grad_fn = linear(x_t, self.w_in, self.b_in, self._is_training)

            # hidden projection W_h * h_t-1 + b_h
            h_prev = h[:, t - 1]
            h_r, h_r_grad_fn = linear(h_prev, self.w_hr, self.b_hr, self._is_training)
            h_z, h_z_grad_fn = linear(h_prev, self.w_hz, self.b_hz, self._is_training)
            h_n[:, t], h_n_grad_fn = linear(
                h_prev, self.w_hn, self.b_hn, self._is_training
            )

            # gates
            r[:, t], r_grad_fn = sigmoid(x_r + h_r, self._is_training)  # reset gate
            z[:, t], z_grad_fn = sigmoid(x_z + h_z, self._is_training)  # update gate

            # candidate hidden state n_t = act(x_n + r_t * h_t-1)
            n[:, t], n_grad_fn = self.act(x_n + r[:, t] * h_n[:, t], self._is_training)

            # hidden state h_t = (1 - z_t) * n_t + z_t * h_t-1
            h[:, t] = (1 - z[:, t]) * n[:, t] + z[:, t] * h[:, t - 1]

            # remember gradient functions
            if self._is_training:
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

        if self._is_training:

            def _backward(dy: Tensor) -> Tensor:
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
                    if t > 0:
                        update_parameter_grad(self.w_hr, dw_hr)
                    update_parameter_grad(self.b_hr, db_hr)

                    dh_z, dw_hz, db_hz = h_z_grad_fn(dz_preact)
                    if t > 0:
                        update_parameter_grad(self.w_hz, dw_hz)
                    update_parameter_grad(self.b_hz, db_hz)

                    dh_n, dw_hn, db_hn = h_n_grad_fn(r[:, t] * dn_preact)
                    if t > 0:
                        update_parameter_grad(self.w_hn, dw_hn)
                    update_parameter_grad(self.b_hn, db_hn)

                    dh += dh_r + dh_z + dh_n

                    # input projection gradients
                    dx_r, dw_ir, db_ir = x_r_grad_fn(dr_preact)
                    update_parameter_grad(self.w_ir, dw_ir)
                    update_parameter_grad(self.b_ir, db_ir)

                    dx_z, dw_iz, db_iz = x_z_grad_fn(dz_preact)
                    update_parameter_grad(self.w_iz, dw_iz)
                    update_parameter_grad(self.b_iz, db_iz)

                    dx_n, dw_in, db_in = x_n_grad_fn(dn_preact)
                    update_parameter_grad(self.w_in, dw_in)
                    update_parameter_grad(self.b_in, db_in)

                    dx[:, t] = dx_r + dx_z + dx_n
                return dx

            self._backward = _backward

        return h if self.return_sequence else h[:, -1]
