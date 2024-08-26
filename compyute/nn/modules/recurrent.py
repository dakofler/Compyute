"""Neural network recurrent modules."""

from typing import Literal, Optional

from ...tensor_ops.creating import empty, empty_like, zeros, zeros_like
from ...tensors import Tensor
from ...typing import DType
from ..functional.activations import FReLU, FSigmoid, FTanh
from ..functional.functions import FunctionCache
from ..functional.linear import FLinear
from ..parameter import Parameter, update_parameter_grad
from ..utils.initializers import XavierUniform, Zeros
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
        self.act = FReLU if activation == "relu" else FTanh
        self.return_sequence = return_sequence
        self._fcaches: list[FunctionCache] = []

        # init input parameters
        self.w_i = Parameter(empty((h_channels, in_channels), dtype=dtype))
        self.b_i = Parameter(empty((h_channels,), dtype=dtype)) if bias else None

        # init hidden parameters
        self.w_h = Parameter(empty((h_channels, h_channels), dtype=dtype))
        self.b_h = Parameter(empty((h_channels,), dtype=dtype)) if bias else None

        self._init_parameters_and_buffers()

    def _init_parameters_and_buffers(self) -> None:
        XavierUniform()(self.w_i, self.w_h)
        if self.bias:
            Zeros()(self.b_i, self.b_h)

    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [3])

        # input projection
        x_h = FLinear.forward(self._fcache, x, self.w_i, self.b_i)

        # iterate over timesteps
        h = zeros((*x.shape[:2], self.h_channels), x.device, x.dtype)
        for t in range(x.shape[1]):
            fcache = FunctionCache()
            self._fcaches.append(fcache)

            # hidden projection
            h_h = FLinear.forward(fcache, h[:, t - 1], self.w_h, self.b_h)

            # apply non-linearity
            h[:, t] = self.act.forward(fcache, x_h[:, t] + h_h)

        y = h if self.return_sequence else h[:, -1]

        self._fcache.rec_x = x
        return y

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        x = self._fcache.rec_x
        dh = 0

        # iterate backwards over timesteps
        dact = zeros((*x.shape[:2], self.h_channels), x.device, x.dtype)
        for t in range(x.shape[1] - 1, -1, -1):
            cache = self._fcaches[t]

            # add output gradients if returning sequence or last time step
            if self.return_sequence:
                dh += dy[:, t]
            elif t == x.shape[1] - 1:
                dh += dy

            # non-linearity backward
            dact[:, t] = self.act.backward(cache, dh)

            # hidden projection backward
            dh, dw_h, db_h = FLinear.backward(cache, dact[:, t])
            update_parameter_grad(self.w_h, dw_h)
            update_parameter_grad(self.b_h, db_h)

        # input projeciton backward
        dx, dw_i, db_i = FLinear.backward(self._fcache, dact)
        update_parameter_grad(self.w_i, dw_i)
        update_parameter_grad(self.b_i, db_i)

        return dx


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
        self.act = FReLU if activation == "relu" else FTanh
        self.return_sequence = return_sequence
        self._ii_fcache = FunctionCache()
        self._if_fcache = FunctionCache()
        self._ig_fcache = FunctionCache()
        self._io_fcache = FunctionCache()
        self._fcaches: list[tuple[FunctionCache, ...]] = []

        # init input parameters
        self.w_ii = Parameter(empty((h_channels, in_channels), dtype=dtype))
        self.b_ii = Parameter(empty((h_channels,), dtype=dtype)) if bias else None
        self.w_if = Parameter(empty((h_channels, in_channels), dtype=dtype))
        self.b_if = Parameter(empty((h_channels,), dtype=dtype)) if bias else None
        self.w_ig = Parameter(empty((h_channels, in_channels), dtype=dtype))
        self.b_ig = Parameter(empty((h_channels,), dtype=dtype)) if bias else None
        self.w_io = Parameter(empty((h_channels, in_channels), dtype=dtype))
        self.b_io = Parameter(empty((h_channels,), dtype=dtype)) if bias else None

        # init hidden parameters
        self.w_hi = Parameter(empty((h_channels, h_channels), dtype=dtype))
        self.b_hi = Parameter(empty((h_channels,), dtype=dtype)) if bias else None
        self.w_hf = Parameter(empty((h_channels, h_channels), dtype=dtype))
        self.b_hf = Parameter(empty((h_channels,), dtype=dtype)) if bias else None
        self.w_hg = Parameter(empty((h_channels, h_channels), dtype=dtype))
        self.b_hg = Parameter(empty((h_channels,), dtype=dtype)) if bias else None
        self.w_ho = Parameter(empty((h_channels, h_channels), dtype=dtype))
        self.b_ho = Parameter(empty((h_channels,), dtype=dtype)) if bias else None

        self._init_parameters_and_buffers()

    def _init_parameters_and_buffers(self) -> None:
        XavierUniform()(
            self.w_ii,
            self.w_if,
            self.w_ig,
            self.w_io,
            self.w_hi,
            self.w_hf,
            self.w_hg,
            self.w_ho,
        )
        if self.bias:
            Zeros()(
                self.b_ii,
                self.b_if,
                self.b_ig,
                self.b_io,
                self.b_hi,
                self.b_hf,
                self.b_hg,
                self.b_ho,
            )

    def forward(self, x: Tensor):
        validate_input_axes(self, x, [3])

        i = empty((*x.shape[:2], self.h_channels), x.device, x.dtype)
        f, g, o, c = empty_like(i), empty_like(i), empty_like(i), zeros_like(i)
        act_c, h = empty_like(i), zeros_like(i)

        # input projection W_i * x_t + b_i
        x_i = FLinear.forward(self._ii_fcache, x, self.w_ii, self.b_ii)
        x_f = FLinear.forward(self._if_fcache, x, self.w_if, self.b_if)
        x_g = FLinear.forward(self._ig_fcache, x, self.w_ig, self.b_ig)
        x_o = FLinear.forward(self._io_fcache, x, self.w_io, self.b_io)

        # iterate over timesteps
        for t in range(x.shape[1]):
            i_fcache = FunctionCache()
            f_fcache = FunctionCache()
            g_fcache = FunctionCache()
            o_fcache = FunctionCache()
            c_fcache = FunctionCache()
            self._fcaches.append((i_fcache, f_fcache, g_fcache, o_fcache, c_fcache))

            # hidden projection W_h * h_t-1 + b_h
            h_i = FLinear.forward(i_fcache, h[:, t - 1], self.w_hi, self.b_hi)
            h_f = FLinear.forward(f_fcache, h[:, t - 1], self.w_hf, self.b_hf)
            h_g = FLinear.forward(g_fcache, h[:, t - 1], self.w_hg, self.b_hg)
            h_o = FLinear.forward(o_fcache, h[:, t - 1], self.w_ho, self.b_ho)

            # gates
            i[:, t] = FSigmoid.forward(i_fcache, x_i[:, t] + h_i)  # input gate
            f[:, t] = FSigmoid.forward(f_fcache, x_f[:, t] + h_f)  # forget gate
            o[:, t] = FSigmoid.forward(o_fcache, x_o[:, t] + h_o)  # output gate

            # input node
            g[:, t] = self.act.forward(g_fcache, x_g[:, t] + h_g)

            # carry state c_t = f_t * c_t-1 + i_t * g_t
            c[:, t] = f[:, t] * c[:, t - 1] + i[:, t] * g[:, t]

            # memory state h_t = o_t * act(c_t)
            act_c[:, t] = self.act.forward(c_fcache, c[:, t])
            h[:, t] = o[:, t] * act_c[:, t]

        y = h if self.return_sequence else h[:, -1]

        self._fcache.lstm_x = x
        self._fcache.lstm_i = i
        self._fcache.lstm_f = f
        self._fcache.lstm_g = g
        self._fcache.lstm_o = o
        self._fcache.lstm_c = c
        self._fcache.lstm_act_c = act_c
        return y

    def backward(self, dy: Tensor) -> Tensor:
        x = self._fcache.lstm_x
        i = self._fcache.lstm_i
        f = self._fcache.lstm_f
        g = self._fcache.lstm_g
        o = self._fcache.lstm_o
        c = self._fcache.lstm_c
        act_c = self._fcache.lstm_act_c
        dc = zeros_like(c)
        dh = 0
        di_preact = empty_like(i)
        df_preact = empty_like(f)
        dg_preact = empty_like(g)
        do_preact = empty_like(o)

        # iterate backwards over timesteps
        for t in range(x.shape[1] - 1, -1, -1):
            i_fcache, f_fcache, g_fcache, o_fcache, c_fcache = self._fcaches[t]

            # add output gradients if returning sequence or last time step
            if self.return_sequence:
                dh += dy[:, t]
            elif t == x.shape[1] - 1:
                dh += dy

            # memory state gradients
            do = act_c[:, t] * dh
            dc[:, t] += self.act.backward(c_fcache, dh) * o[:, t]

            # carry state gradients
            df = c[:, t - 1] * dc[:, t] if t > 0 else 0
            if t > 0:
                dc[:, t - 1] += f[:, t] * dc[:, t]
            di = g[:, t] * dc[:, t]
            dg = i[:, t] * dc[:, t]

            # gate gradients
            di_preact[:, t] = FSigmoid.backward(i_fcache, di)
            df_preact[:, t] = FSigmoid.backward(f_fcache, df)
            do_preact[:, t] = FSigmoid.backward(o_fcache, do)

            # input node gradients
            dg_preact[:, t] = self.act.backward(g_fcache, dg)

            # hidden projection gradients
            dh_i, dw_hi, db_hi = FLinear.backward(i_fcache, di_preact[:, t])
            if t > 0:
                update_parameter_grad(self.w_hi, dw_hi)
            update_parameter_grad(self.b_hi, db_hi)

            dh_f, dw_hf, db_hf = FLinear.backward(f_fcache, df_preact[:, t])
            if t > 0:
                update_parameter_grad(self.w_hf, dw_hf)
            update_parameter_grad(self.b_hf, db_hf)

            dh_g, dw_hg, db_hg = FLinear.backward(g_fcache, dg_preact[:, t])
            if t > 0:
                update_parameter_grad(self.w_hg, dw_hg)
            update_parameter_grad(self.b_hg, db_hg)

            dh_o, dw_ho, db_ho = FLinear.backward(o_fcache, do_preact[:, t])
            if t > 0:
                update_parameter_grad(self.w_ho, dw_ho)
            update_parameter_grad(self.b_ho, db_ho)

            dh = dh_i + dh_f + dh_g + dh_o

        # input projection gradients
        dx_i, dw_ii, db_ii = FLinear.backward(self._ii_fcache, di_preact)
        update_parameter_grad(self.w_ii, dw_ii)
        update_parameter_grad(self.b_ii, db_ii)

        dx_f, dw_if, db_if = FLinear.backward(self._if_fcache, df_preact)
        update_parameter_grad(self.w_if, dw_if)
        update_parameter_grad(self.b_if, db_if)

        dx_g, dw_ig, db_ig = FLinear.backward(self._ig_fcache, dg_preact)
        update_parameter_grad(self.w_ig, dw_ig)
        update_parameter_grad(self.b_ig, db_ig)

        dx_o, dw_io, db_io = FLinear.backward(self._io_fcache, do_preact)
        update_parameter_grad(self.w_io, dw_io)
        update_parameter_grad(self.b_io, db_io)

        return dx_i + dx_f + dx_g + dx_o


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
        self.act = FReLU if activation == "relu" else FTanh
        self.return_sequence = return_sequence
        self._ir_fcache = FunctionCache()
        self._iz_fcache = FunctionCache()
        self._in_fcache = FunctionCache()
        self._fcaches: list[tuple[FunctionCache, ...]] = []

        # init input parameters
        self.w_ir = Parameter(empty((h_channels, in_channels), dtype=dtype))
        self.b_ir = Parameter(empty((h_channels,), dtype=dtype)) if bias else None
        self.w_iz = Parameter(empty((h_channels, in_channels), dtype=dtype))
        self.b_iz = Parameter(empty((h_channels,), dtype=dtype)) if bias else None
        self.w_in = Parameter(empty((h_channels, in_channels), dtype=dtype))
        self.b_in = Parameter(empty((h_channels,), dtype=dtype)) if bias else None

        # init hidden parameters
        self.w_hr = Parameter(empty((h_channels, h_channels), dtype=dtype))
        self.b_hr = Parameter(empty((h_channels,), dtype=dtype)) if bias else None
        self.w_hz = Parameter(empty((h_channels, h_channels), dtype=dtype))
        self.b_hz = Parameter(empty((h_channels,), dtype=dtype)) if bias else None
        self.w_hn = Parameter(empty((h_channels, h_channels), dtype=dtype))
        self.b_hn = Parameter(empty((h_channels,), dtype=dtype)) if bias else None

        self._init_parameters_and_buffers()

    def _init_parameters_and_buffers(self) -> None:
        XavierUniform()(
            self.w_ir, self.w_iz, self.w_in, self.w_hr, self.w_hz, self.w_hn
        )
        if self.bias:
            Zeros()(self.b_ir, self.b_iz, self.b_in, self.b_hr, self.b_hz, self.b_hn)

    def forward(self, x: Tensor):
        validate_input_axes(self, x, [3])

        r = empty((*x.shape[:2], self.h_channels), x.device, x.dtype)
        z, n, h_n, h = empty_like(r), empty_like(r), empty_like(r), zeros_like(r)

        # input projection W_i * x_t + b_i
        x_r = FLinear.forward(self._ir_fcache, x, self.w_ir, self.b_ir)
        x_z = FLinear.forward(self._iz_fcache, x, self.w_iz, self.b_iz)
        x_n = FLinear.forward(self._in_fcache, x, self.w_in, self.b_in)

        # iterate over timesteps
        for t in range(x.shape[1]):
            r_fcache = FunctionCache()
            z_fcache = FunctionCache()
            n_fcache = FunctionCache()
            self._fcaches.append((r_fcache, z_fcache, n_fcache))

            # hidden projection W_h * h_t-1 + b_h
            h_r = FLinear.forward(r_fcache, h[:, t - 1], self.w_hr, self.b_hr)
            h_z = FLinear.forward(z_fcache, h[:, t - 1], self.w_hz, self.b_hz)
            h_n[:, t] = FLinear.forward(n_fcache, h[:, t - 1], self.w_hn, self.b_hn)

            # gates
            r[:, t] = FSigmoid.forward(r_fcache, x_r[:, t] + h_r)  # reset gate
            z[:, t] = FSigmoid.forward(z_fcache, x_z[:, t] + h_z)  # update gate

            # candidate hidden state n_t = act(x_n + r_t * h_t-1)
            n[:, t] = self.act.forward(n_fcache, x_n[:, t] + r[:, t] * h_n[:, t])

            # hidden state h_t = (1 - z_t) * n_t + z_t * h_t-1
            h[:, t] = (1 - z[:, t]) * n[:, t] + z[:, t] * h[:, t - 1]

        y = h if self.return_sequence else h[:, -1]

        self._fcache.gru_x = x
        self._fcache.gru_r = r
        self._fcache.gru_z = z
        self._fcache.gru_n = n
        self._fcache.gru_h_n = h_n
        self._fcache.gru_h = h
        return y

    def backward(self, dy: Tensor) -> Tensor:
        x = self._fcache.gru_x
        r = self._fcache.gru_r
        z = self._fcache.gru_z
        n = self._fcache.gru_n
        h_n = self._fcache.gru_h_n
        h = self._fcache.gru_h
        dh = 0
        dr_preact = empty_like(r)
        dz_preact = empty_like(z)
        dn_preact = empty_like(n)

        # iterate backwards over timesteps
        for t in range(x.shape[1] - 1, -1, -1):
            r_fcache, z_fcache, n_fcache = self._fcaches[t]

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
            dn_preact[:, t] = self.act.backward(n_fcache, dn)
            dr = h_n[:, t] * dn_preact[:, t]

            # gate gradients
            dr_preact[:, t] = FSigmoid.backward(r_fcache, dr)
            dz_preact[:, t] = FSigmoid.backward(z_fcache, dz)

            # hidden projection gradients
            dh_r, dw_hr, db_hr = FLinear.backward(r_fcache, dr_preact[:, t])
            if t > 0:
                update_parameter_grad(self.w_hr, dw_hr)
            update_parameter_grad(self.b_hr, db_hr)

            dh_z, dw_hz, db_hz = FLinear.backward(z_fcache, dz_preact[:, t])
            if t > 0:
                update_parameter_grad(self.w_hz, dw_hz)
            update_parameter_grad(self.b_hz, db_hz)

            dh_n, dw_hn, db_hn = FLinear.backward(n_fcache, r[:, t] * dn_preact[:, t])
            if t > 0:
                update_parameter_grad(self.w_hn, dw_hn)
            update_parameter_grad(self.b_hn, db_hn)

            dh += dh_r + dh_z + dh_n

        # input projection gradients
        dx_r, dw_ir, db_ir = FLinear.backward(self._ir_fcache, dr_preact)
        update_parameter_grad(self.w_ir, dw_ir)
        update_parameter_grad(self.b_ir, db_ir)

        dx_z, dw_iz, db_iz = FLinear.backward(self._iz_fcache, dz_preact)
        update_parameter_grad(self.w_iz, dw_iz)
        update_parameter_grad(self.b_iz, db_iz)

        dx_n, dw_in, db_in = FLinear.backward(self._in_fcache, dn_preact)
        update_parameter_grad(self.w_in, dw_in)
        update_parameter_grad(self.b_in, db_in)

        return dx_r + dx_z + dx_n
