"""Neural network recurrent modules."""

from typing import Literal, Optional

from ...tensor_ops.creating import empty, empty_like, zeros, zeros_like
from ...tensors import Tensor
from ...typing import DType
from ..functional.activations import FReLU, FSigmoid, FTanh
from ..functional.linear import FLinear
from ..parameter import Parameter, update_parameter_grad
from ..utils.initializers import init_xavier_uniform, init_zeros
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

        # init input parameters
        self.w_i = Parameter(empty((h_channels, in_channels), dtype=dtype))
        self.b_i = None if not bias else Parameter(empty((h_channels,), dtype=dtype))

        # init hidden parameters
        self.w_h = Parameter(empty((h_channels, h_channels), dtype=dtype))
        self.b_h = None if not bias else Parameter(empty((h_channels,), dtype=dtype))

        self._init_parameters_and_buffers()

    def _init_parameters_and_buffers(self) -> None:
        init_xavier_uniform(self.w_i, self.w_h)
        if self.bias:
            init_zeros(self.b_i, self.b_h)

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [3])

        # input projection
        x_h = FLinear.forward(self.fcache, x, self.w_i, self.b_i)

        # iterate over timesteps
        h_shape = (*x.shape[:2], self.h_channels)
        h = zeros(h_shape, device=x.device, dtype=x.dtype)
        for t in range(x.shape[1]):

            # hidden projection
            h_h = FLinear.forward(self.fcache, h[:, t - 1], self.w_h, self.b_h)

            # apply non-linearity
            h[:, t] = self.act.forward(self.fcache, x_h[:, t] + h_h)

        y = h if self.return_sequence else h[:, -1]

        self.fcache.x_shape, self.fcache.h_shape = x.shape, h_shape
        return y

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        x_shape, h_shape = self.fcache.x_shape, self.fcache.h_shape
        dpreact = zeros(h_shape, device=dy.device, dtype=dy.dtype)
        dh = 0  # TODO: should be a tensor

        # iterate backwards over timesteps
        for t in range(x_shape[1] - 1, -1, -1):

            # add output gradients if returning sequence or last time step
            if self.return_sequence:
                dh += dy[:, t]
            elif t == x_shape[1] - 1:
                dh += dy

            # non-linearity backward
            dpreact[:, t] = self.act.backward(self.fcache, dh)

            # hidden projection backward
            dh, dw_h, db_h = FLinear.backward(self.fcache, dpreact[:, t])
            update_parameter_grad(self.w_h, dw_h)
            update_parameter_grad(self.b_h, db_h)

        # input projeciton backward
        dx, dw_i, db_i = FLinear.backward(self.fcache, dpreact)
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

        # init input parameters
        self.w_ii = Parameter(empty((h_channels, in_channels), dtype=dtype))
        self.b_ii = None if not bias else Parameter(empty((h_channels,), dtype=dtype))
        self.w_if = Parameter(empty((h_channels, in_channels), dtype=dtype))
        self.b_if = None if not bias else Parameter(empty((h_channels,), dtype=dtype))
        self.w_ig = Parameter(empty((h_channels, in_channels), dtype=dtype))
        self.b_ig = None if not bias else Parameter(empty((h_channels,), dtype=dtype))
        self.w_io = Parameter(empty((h_channels, in_channels), dtype=dtype))
        self.b_io = None if not bias else Parameter(empty((h_channels,), dtype=dtype))

        # init hidden parameters
        self.w_hi = Parameter(empty((h_channels, h_channels), dtype=dtype))
        self.b_hi = None if not bias else Parameter(empty((h_channels,), dtype=dtype))
        self.w_hf = Parameter(empty((h_channels, h_channels), dtype=dtype))
        self.b_hf = None if not bias else Parameter(empty((h_channels,), dtype=dtype))
        self.w_hg = Parameter(empty((h_channels, h_channels), dtype=dtype))
        self.b_hg = None if not bias else Parameter(empty((h_channels,), dtype=dtype))
        self.w_ho = Parameter(empty((h_channels, h_channels), dtype=dtype))
        self.b_ho = None if not bias else Parameter(empty((h_channels,), dtype=dtype))

        self._init_parameters_and_buffers()

    def _init_parameters_and_buffers(self) -> None:
        init_xavier_uniform(
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
            init_zeros(
                self.b_ii,
                self.b_if,
                self.b_ig,
                self.b_io,
                self.b_hi,
                self.b_hf,
                self.b_hg,
                self.b_ho,
            )

    @Module.register_forward
    def forward(self, x: Tensor):
        validate_input_axes(self, x, [3])
        i = empty((*x.shape[:2], self.h_channels), device=x.device, dtype=x.dtype)
        f, g, o, c = empty_like(i), empty_like(i), empty_like(i), zeros_like(i)
        act_c, h = empty_like(i), zeros_like(i)

        # input projection W_i * x_t + b_i
        x_i = FLinear.forward(self.fcache, x, self.w_ii, self.b_ii)
        x_f = FLinear.forward(self.fcache, x, self.w_if, self.b_if)
        x_g = FLinear.forward(self.fcache, x, self.w_ig, self.b_ig)
        x_o = FLinear.forward(self.fcache, x, self.w_io, self.b_io)

        # iterate over timesteps
        for t in range(x.shape[1]):

            # hidden projection W_h * h_t-1 + b_h
            h_i = FLinear.forward(self.fcache, h[:, t - 1], self.w_hi, self.b_hi)
            h_f = FLinear.forward(self.fcache, h[:, t - 1], self.w_hf, self.b_hf)
            h_g = FLinear.forward(self.fcache, h[:, t - 1], self.w_hg, self.b_hg)
            h_o = FLinear.forward(self.fcache, h[:, t - 1], self.w_ho, self.b_ho)

            # gates
            i[:, t] = FSigmoid.forward(self.fcache, x_i[:, t] + h_i)  # input gate
            f[:, t] = FSigmoid.forward(self.fcache, x_f[:, t] + h_f)  # forget gate
            o[:, t] = FSigmoid.forward(self.fcache, x_o[:, t] + h_o)  # output gate

            # input node
            g[:, t] = self.act.forward(self.fcache, x_g[:, t] + h_g)

            # carry state c_t = f_t * c_t-1 + i_t * g_t
            c[:, t] = f[:, t] * c[:, t - 1] + i[:, t] * g[:, t]

            # memory state h_t = o_t * act(c_t)
            act_c[:, t] = self.act.forward(self.fcache, c[:, t])
            h[:, t] = o[:, t] * act_c[:, t]

        y = h if self.return_sequence else h[:, -1]

        self.fcache.i, self.fcache.f, self.fcache.g, self.fcache.o = i, f, g, o
        self.fcache.x_shape, self.fcache.c, self.fcache.act_c = x.shape, c, act_c
        return y

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        i, f, g, o = self.fcache.i, self.fcache.f, self.fcache.g, self.fcache.o
        x_shape, c, act_c = self.fcache.x_shape, self.fcache.c, self.fcache.act_c
        di_preact, df_preact, dg_preact = empty_like(i), empty_like(f), empty_like(g)
        do_preact, dc = empty_like(o), zeros_like(c)
        dh = 0

        # iterate backwards over timesteps
        for t in range(x_shape[1] - 1, -1, -1):

            # add output gradients if returning sequence or last time step
            if self.return_sequence:
                dh += dy[:, t]
            elif t == x_shape[1] - 1:
                dh += dy

            # memory state gradients
            do = act_c[:, t] * dh
            dc[:, t] += self.act.backward(self.fcache, dh) * o[:, t]

            # carry state gradients
            df = 0 if t < 1 else c[:, t - 1] * dc[:, t]
            if t > 0:
                dc[:, t - 1] += f[:, t] * dc[:, t]
            di = g[:, t] * dc[:, t]
            dg = i[:, t] * dc[:, t]

            # input node gradients
            dg_preact[:, t] = self.act.backward(self.fcache, dg)

            # gate gradients
            do_preact[:, t] = FSigmoid.backward(self.fcache, do)
            df_preact[:, t] = FSigmoid.backward(self.fcache, df)
            di_preact[:, t] = FSigmoid.backward(self.fcache, di)

            # hidden projection gradients
            dh_o, dw_ho, db_ho = FLinear.backward(self.fcache, do_preact[:, t])
            dh_g, dw_hg, db_hg = FLinear.backward(self.fcache, dg_preact[:, t])
            dh_f, dw_hf, db_hf = FLinear.backward(self.fcache, df_preact[:, t])
            dh_i, dw_hi, db_hi = FLinear.backward(self.fcache, di_preact[:, t])

            if t > 0:
                update_parameter_grad(self.w_ho, dw_ho)
                update_parameter_grad(self.w_hg, dw_hg)
                update_parameter_grad(self.w_hf, dw_hf)
                update_parameter_grad(self.w_hi, dw_hi)

            update_parameter_grad(self.b_ho, db_ho)
            update_parameter_grad(self.b_hg, db_hg)
            update_parameter_grad(self.b_hf, db_hf)
            update_parameter_grad(self.b_hi, db_hi)

            dh = dh_i + dh_f + dh_g + dh_o

        # input projection gradients
        dx_o, dw_io, db_io = FLinear.backward(self.fcache, do_preact)
        dx_g, dw_ig, db_ig = FLinear.backward(self.fcache, dg_preact)
        dx_f, dw_if, db_if = FLinear.backward(self.fcache, df_preact)
        dx_i, dw_ii, db_ii = FLinear.backward(self.fcache, di_preact)

        update_parameter_grad(self.w_io, dw_io)
        update_parameter_grad(self.b_io, db_io)
        update_parameter_grad(self.w_ig, dw_ig)
        update_parameter_grad(self.b_ig, db_ig)
        update_parameter_grad(self.w_if, dw_if)
        update_parameter_grad(self.b_if, db_if)
        update_parameter_grad(self.w_ii, dw_ii)
        update_parameter_grad(self.b_ii, db_ii)

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

        # init input parameters
        self.w_ir = Parameter(empty((h_channels, in_channels), dtype=dtype))
        self.b_ir = None if not bias else Parameter(empty((h_channels,), dtype=dtype))
        self.w_iz = Parameter(empty((h_channels, in_channels), dtype=dtype))
        self.b_iz = None if not bias else Parameter(empty((h_channels,), dtype=dtype))
        self.w_in = Parameter(empty((h_channels, in_channels), dtype=dtype))
        self.b_in = None if not bias else Parameter(empty((h_channels,), dtype=dtype))

        # init hidden parameters
        self.w_hr = Parameter(empty((h_channels, h_channels), dtype=dtype))
        self.b_hr = None if not bias else Parameter(empty((h_channels,), dtype=dtype))
        self.w_hz = Parameter(empty((h_channels, h_channels), dtype=dtype))
        self.b_hz = None if not bias else Parameter(empty((h_channels,), dtype=dtype))
        self.w_hn = Parameter(empty((h_channels, h_channels), dtype=dtype))
        self.b_hn = None if not bias else Parameter(empty((h_channels,), dtype=dtype))

        self._init_parameters_and_buffers()

    def _init_parameters_and_buffers(self) -> None:
        init_xavier_uniform(
            self.w_ir, self.w_iz, self.w_in, self.w_hr, self.w_hz, self.w_hn
        )
        if self.bias:
            init_zeros(self.b_ir, self.b_iz, self.b_in, self.b_hr, self.b_hz, self.b_hn)

    @Module.register_forward
    def forward(self, x: Tensor):
        validate_input_axes(self, x, [3])
        r = empty((*x.shape[:2], self.h_channels), device=x.device, dtype=x.dtype)
        z, n, h_n, h = empty_like(r), empty_like(r), empty_like(r), zeros_like(r)

        # input projection W_i * x_t + b_i
        x_r = FLinear.forward(self.fcache, x, self.w_ir, self.b_ir)
        x_z = FLinear.forward(self.fcache, x, self.w_iz, self.b_iz)
        x_n = FLinear.forward(self.fcache, x, self.w_in, self.b_in)

        # iterate over timesteps
        for t in range(x.shape[1]):

            # hidden projection W_h * h_t-1 + b_h
            h_r = FLinear.forward(self.fcache, h[:, t - 1], self.w_hr, self.b_hr)
            h_z = FLinear.forward(self.fcache, h[:, t - 1], self.w_hz, self.b_hz)
            h_n[:, t] = FLinear.forward(self.fcache, h[:, t - 1], self.w_hn, self.b_hn)

            # gates
            r[:, t] = FSigmoid.forward(self.fcache, x_r[:, t] + h_r)  # reset gate
            z[:, t] = FSigmoid.forward(self.fcache, x_z[:, t] + h_z)  # update gate

            # candidate hidden state n_t = act(x_n + r_t * h_t-1)
            n[:, t] = self.act.forward(self.fcache, x_n[:, t] + r[:, t] * h_n[:, t])

            # hidden state h_t = (1 - z_t) * n_t + z_t * h_t-1
            h[:, t] = (1 - z[:, t]) * n[:, t] + z[:, t] * h[:, t - 1]

        y = h if self.return_sequence else h[:, -1]

        self.fcache.r, self.fcache.z, self.fcache.n = r, z, n
        self.fcache.x_shape, self.fcache.h_n, self.fcache.h = x.shape, h_n, h
        return y

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        r, z, n = self.fcache.r, self.fcache.z, self.fcache.n
        x_shape, h_n, h = self.fcache.x_shape, self.fcache.h_n, self.fcache.h
        dr_preact, dz_preact, dn_preact = empty_like(r), empty_like(z), empty_like(n)
        dh = 0

        # iterate backwards over timesteps
        for t in range(x_shape[1] - 1, -1, -1):

            # add output gradients if returning sequence or last time step
            if self.return_sequence:
                dh += dy[:, t]
            elif t == x_shape[1] - 1:
                dh += dy

            # hidden state gradients
            dz = ((0 if t < 1 else h[:, t - 1]) - n[:, t]) * dh
            dn = (1 - z[:, t]) * dh
            dh = z[:, t] * dh

            # candidate hidden state gradients
            dn_preact[:, t] = self.act.backward(self.fcache, dn)
            dr = h_n[:, t] * dn_preact[:, t]

            # gate gradients
            dz_preact[:, t] = FSigmoid.backward(self.fcache, dz)
            dr_preact[:, t] = FSigmoid.backward(self.fcache, dr)

            # hidden projection gradients
            r_dn_preact = r[:, t] * dn_preact[:, t]
            dh_n, dw_hn, db_hn = FLinear.backward(self.fcache, r_dn_preact)
            dh_z, dw_hz, db_hz = FLinear.backward(self.fcache, dz_preact[:, t])
            dh_r, dw_hr, db_hr = FLinear.backward(self.fcache, dr_preact[:, t])

            if t > 0:
                update_parameter_grad(self.w_hn, dw_hn)
                update_parameter_grad(self.w_hz, dw_hz)
                update_parameter_grad(self.w_hr, dw_hr)

            update_parameter_grad(self.b_hn, db_hn)
            update_parameter_grad(self.b_hz, db_hz)
            update_parameter_grad(self.b_hr, db_hr)

            dh += dh_r + dh_z + dh_n

        # input projection gradients
        dx_n, dw_in, db_in = FLinear.backward(self.fcache, dn_preact)
        dx_z, dw_iz, db_iz = FLinear.backward(self.fcache, dz_preact)
        dx_r, dw_ir, db_ir = FLinear.backward(self.fcache, dr_preact)

        update_parameter_grad(self.w_in, dw_in)
        update_parameter_grad(self.b_in, db_in)
        update_parameter_grad(self.w_iz, dw_iz)
        update_parameter_grad(self.b_iz, db_iz)
        update_parameter_grad(self.w_ir, dw_ir)
        update_parameter_grad(self.b_ir, db_ir)

        return dx_r + dx_z + dx_n
