"""Neural network recurrent modules."""

import math
from typing import Literal, Optional

from ...random import uniform
from ...tensors import Tensor
from ...typing import DType
from ..functional.recurrent_funcs import GRUFn, LSTMFn, RecurrentFn
from ..parameter import Parameter, update_parameter_grad
from .module import Module

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
        - :math:`B` ... batch dimension
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
    dtype : DType, optional
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights and biases are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{h}}}`.
    """

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        activation: str = "tanh",
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.bias = bias
        self.activation = activation

        # init parameters
        k = 1.0 / math.sqrt(self.h_channels)
        w_i_init = lambda: uniform((h_channels, in_channels), -k, k, dtype=dtype)
        w_h_init = lambda: uniform((h_channels, h_channels), -k, k, dtype=dtype)
        b_init = lambda: uniform((h_channels,), -k, k, dtype=dtype)

        self.w_i = Parameter(w_i_init())
        self.b_i = Parameter(b_init())

        self.w_h = Parameter(w_h_init())
        self.b_h = Parameter(b_init())

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return RecurrentFn.forward(
            self.fcache, x, self.w_i, self.b_i, self.w_h, self.b_h, self.activation
        )

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw_i, db_i, dw_h, db_h = RecurrentFn.backward(self.fcache, dy)
        update_parameter_grad(self.w_i, dw_i)
        update_parameter_grad(self.b_i, db_i)
        update_parameter_grad(self.w_h, dw_h)
        update_parameter_grad(self.b_h, db_h)
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
        - :math:`B` ... batch dimension
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
    dtype : DType, optional
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights and biases are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{h}}}`.
    """

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        activation: Literal["relu", "tanh"] = "tanh",
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.bias = bias
        self.activation = activation

        # init parameters
        k = 1.0 / math.sqrt(self.h_channels)
        w_i_init = lambda: uniform((h_channels, in_channels), -k, k, dtype=dtype)
        w_h_init = lambda: uniform((h_channels, h_channels), -k, k, dtype=dtype)
        b_init = lambda: uniform((h_channels,), -k, k, dtype=dtype)

        self.w_ii = Parameter(w_i_init())
        self.b_ii = None if not bias else Parameter(b_init())
        self.w_if = Parameter(w_i_init())
        self.b_if = None if not bias else Parameter(b_init())
        self.w_ig = Parameter(w_i_init())
        self.b_ig = None if not bias else Parameter(b_init())
        self.w_io = Parameter(w_i_init())
        self.b_io = None if not bias else Parameter(b_init())

        self.w_hi = Parameter(w_h_init())
        self.b_hi = None if not bias else Parameter(b_init())
        self.w_hf = Parameter(w_h_init())
        self.b_hf = None if not bias else Parameter(b_init())
        self.w_hg = Parameter(w_h_init())
        self.b_hg = None if not bias else Parameter(b_init())
        self.w_ho = Parameter(w_h_init())
        self.b_ho = None if not bias else Parameter(b_init())

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return LSTMFn.forward(
            self.fcache,
            x,
            self.w_ii,
            self.b_ii,
            self.w_if,
            self.b_if,
            self.w_ig,
            self.b_ig,
            self.w_io,
            self.b_io,
            self.w_hi,
            self.b_hi,
            self.w_hf,
            self.b_hf,
            self.w_hg,
            self.b_hg,
            self.w_ho,
            self.b_ho,
            self.activation,
        )

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        (
            dx,
            dw_ii,
            db_ii,
            dw_if,
            db_if,
            dw_ig,
            db_ig,
            dw_io,
            db_io,
            dw_hi,
            db_hi,
            dw_hf,
            db_hf,
            dw_hg,
            db_hg,
            dw_ho,
            db_ho,
        ) = LSTMFn.backward(self.fcache, dy)
        update_parameter_grad(self.w_ii, dw_ii)
        update_parameter_grad(self.b_ii, db_ii)
        update_parameter_grad(self.w_if, dw_if)
        update_parameter_grad(self.b_if, db_if)
        update_parameter_grad(self.w_ig, dw_ig)
        update_parameter_grad(self.b_ig, db_ig)
        update_parameter_grad(self.w_io, dw_io)
        update_parameter_grad(self.b_io, db_io)
        update_parameter_grad(self.w_hi, dw_hi)
        update_parameter_grad(self.b_hi, db_hi)
        update_parameter_grad(self.w_hf, dw_hf)
        update_parameter_grad(self.b_hf, db_hf)
        update_parameter_grad(self.w_hg, dw_hg)
        update_parameter_grad(self.b_hg, db_hg)
        update_parameter_grad(self.w_ho, dw_ho)
        update_parameter_grad(self.b_ho, db_ho)
        return dx


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
        - :math:`B` ... batch dimension
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
    dtype : DType, optional
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights and biases are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{h}}}`.
    """

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        activation: Literal["relu", "tanh"] = "tanh",
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.bias = bias
        self.activation = activation

        # init parameters
        k = 1.0 / math.sqrt(self.h_channels)
        w_i_init = lambda: uniform((h_channels, in_channels), -k, k, dtype=dtype)
        w_h_init = lambda: uniform((h_channels, h_channels), -k, k, dtype=dtype)
        b_init = lambda: uniform((h_channels,), -k, k, dtype=dtype)

        self.w_ir = Parameter(w_i_init())
        self.b_ir = None if not bias else Parameter(b_init())
        self.w_iz = Parameter(w_i_init())
        self.b_iz = None if not bias else Parameter(b_init())
        self.w_in = Parameter(w_i_init())
        self.b_in = None if not bias else Parameter(b_init())

        self.w_hr = Parameter(w_h_init())
        self.b_hr = None if not bias else Parameter(b_init())
        self.w_hz = Parameter(w_h_init())
        self.b_hz = None if not bias else Parameter(b_init())
        self.w_hn = Parameter(w_h_init())
        self.b_hn = None if not bias else Parameter(b_init())

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return GRUFn.forward(
            self.fcache,
            x,
            self.w_ir,
            self.b_ir,
            self.w_iz,
            self.b_iz,
            self.w_in,
            self.b_in,
            self.w_hr,
            self.b_hr,
            self.w_hz,
            self.b_hz,
            self.w_hn,
            self.b_hn,
            self.activation,
        )

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        (
            dx,
            dw_ir,
            db_ir,
            dw_iz,
            db_iz,
            dw_in,
            db_in,
            dw_hr,
            db_hr,
            dw_hz,
            db_hz,
            dw_hn,
            db_hn,
        ) = GRUFn.backward(self.fcache, dy)
        update_parameter_grad(self.w_ir, dw_ir)
        update_parameter_grad(self.b_ir, db_ir)
        update_parameter_grad(self.w_iz, dw_iz)
        update_parameter_grad(self.b_iz, db_iz)
        update_parameter_grad(self.w_in, dw_in)
        update_parameter_grad(self.b_in, db_in)
        update_parameter_grad(self.w_hr, dw_hr)
        update_parameter_grad(self.b_hr, db_hr)
        update_parameter_grad(self.w_hz, dw_hz)
        update_parameter_grad(self.b_hz, db_hz)
        update_parameter_grad(self.w_hn, dw_hn)
        update_parameter_grad(self.b_hn, db_hn)
        return dx
