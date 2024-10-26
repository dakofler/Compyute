"""Neural network recurrent functions."""

from typing import Literal, Optional

from ...tensor_ops.creation_ops import empty, empty_like, zeros, zeros_like
from ...tensors import ShapeError, Tensor
from .activation_funcs import ReLUFn, SigmoidFn, TanhFn
from .functions import Function, FunctionCache, PseudoCache
from .linear_funcs import LinearFn

__all__ = ["recurrent", "lstm", "gru"]


class RecurrentFn(Function):
    """Applies the Elman recurrent function to a tensor."""

    @staticmethod
    def forward(
        cache: FunctionCache,
        x: Tensor,
        w_i: Tensor,
        b_i: Optional[Tensor],
        w_h: Tensor,
        b_h: Optional[Tensor],
        activation: str,
    ) -> Tensor:
        if x.ndim != 3:
            raise ShapeError(f"Expected input to be 3D, got {x.ndim}D.")
        if activation not in {"relu", "tanh"}:
            raise ValueError("Activation must be either 'relu' or 'tanh'.")
        act = TanhFn if activation == "tanh" else ReLUFn

        # input projection W_i * x_t + b_i
        x_h = LinearFn.forward(cache, x, w_i, b_i)

        h = zeros_like(x_h)
        for t in range(x.shape[1]):

            # hidden projection W_h * h_t-1 + b_h
            h_h = LinearFn.forward(cache, h[:, t - 1], w_h, b_h)

            # apply activation h_t = act(x_t + h_h)
            h[:, t] = act.forward(cache, x_h[:, t] + h_h)

        cache.push(h.shape, act, b_i is not None)
        return h

    @staticmethod
    def backward(
        cache: FunctionCache, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
        h_shape, act, b = cache.pop()

        B, T, H = h_shape
        dh = zeros((B, H), device=dy.device, dtype=dy.dtype)
        dw_h = zeros((H, H), device=dy.device, dtype=dy.dtype)
        db_h = None if not b else zeros((H,), device=dy.device, dtype=dy.dtype)
        dpreact = empty((B, T, H), device=dy.device, dtype=dy.dtype)

        for t in range(T - 1, -1, -1):

            # activation gradients
            dpreact[:, t] = act.backward(cache, dh + dy[:, t])

            # hidden projection gradients
            dh, dw_h_t, db_h_t = LinearFn.backward(cache, dpreact[:, t])
            if t > 0:
                dw_h += dw_h_t
            if db_h_t:
                db_h += db_h_t

        # input projection gradients
        dx, dw_i, db_i = LinearFn.backward(cache, dpreact)

        return dx, dw_i, db_i, dw_h, db_h


def recurrent(
    x: Tensor,
    w_i: Tensor,
    b_i: Optional[Tensor],
    w_h: Tensor,
    b_h: Optional[Tensor],
    activation: Literal["relu", "tanh"] = "tanh",
) -> Tensor:
    """Applies the Elman recurrent function to a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w_i : Tensor
        Weight tensor for the input projection.
    b_i : Tensor, optional
        Bias tensor for the input projection.
    w_h : Tensor
        Weight tensor for the hidden projection.
    b_h : Tensor, optional
        Bias tensor for the hidden projection.
    activation : Literal["relu", "tanh"], optional
        Activation function to use. Defaults to ``tanh``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.Recurrent`
    """
    return RecurrentFn.forward(PseudoCache(), x, w_i, b_i, w_h, b_h, activation)


class LSTMFn(Function):
    """Applies the LSTM recurrent function to a tensor."""

    @staticmethod
    def forward(
        cache: FunctionCache,
        x: Tensor,
        w_ii: Tensor,
        b_ii: Optional[Tensor],
        w_if: Tensor,
        b_if: Optional[Tensor],
        w_ig: Tensor,
        b_ig: Optional[Tensor],
        w_io: Tensor,
        b_io: Optional[Tensor],
        w_hi: Tensor,
        b_hi: Optional[Tensor],
        w_hf: Tensor,
        b_hf: Optional[Tensor],
        w_hg: Tensor,
        b_hg: Optional[Tensor],
        w_ho: Tensor,
        b_ho: Optional[Tensor],
        activation: str,
    ) -> Tensor:
        if x.ndim != 3:
            raise ShapeError(f"Expected input to be 3D, got {x.ndim}D.")
        if activation not in {"relu", "tanh"}:
            raise ValueError("Activation must be either 'relu' or 'tanh'.")
        act = TanhFn if activation == "tanh" else ReLUFn

        # input projection W_i * x_t + b_i
        x_i = LinearFn.forward(cache, x, w_ii, b_ii)
        x_f = LinearFn.forward(cache, x, w_if, b_if)
        x_g = LinearFn.forward(cache, x, w_ig, b_ig)
        x_o = LinearFn.forward(cache, x, w_io, b_io)

        i, f, g, o = empty_like(x_i), empty_like(x_i), empty_like(x_i), empty_like(x_i)
        c, act_c, h = zeros_like(x_i), empty_like(x_i), zeros_like(x_i)
        for t in range(x.shape[1]):

            # hidden projection W_h * h_t-1 + b_h
            h_i = LinearFn.forward(cache, h[:, t - 1], w_hi, b_hi)
            h_f = LinearFn.forward(cache, h[:, t - 1], w_hf, b_hf)
            h_g = LinearFn.forward(cache, h[:, t - 1], w_hg, b_hg)
            h_o = LinearFn.forward(cache, h[:, t - 1], w_ho, b_ho)

            # gates
            i[:, t] = SigmoidFn.forward(cache, x_i[:, t] + h_i)  # input gate
            f[:, t] = SigmoidFn.forward(cache, x_f[:, t] + h_f)  # forget gate
            o[:, t] = SigmoidFn.forward(cache, x_o[:, t] + h_o)  # output gate

            # candidate cell state
            g[:, t] = act.forward(cache, x_g[:, t] + h_g)

            # cell state c_t = f_t * c_t-1 + i_t * g_t
            c[:, t] = f[:, t] * c[:, t - 1] + i[:, t] * g[:, t]

            # hidden state h_t = o_t * act(c_t)
            act_c[:, t] = act.forward(cache, c[:, t])
            h[:, t] = o[:, t] * act_c[:, t]

        cache.push(i, f, g, o, b_ii is not None, c, act, act_c)
        return h

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> tuple[
        Tensor,
        Tensor,
        Optional[Tensor],
        Tensor,
        Optional[Tensor],
        Tensor,
        Optional[Tensor],
        Tensor,
        Optional[Tensor],
        Tensor,
        Optional[Tensor],
        Tensor,
        Optional[Tensor],
        Tensor,
        Optional[Tensor],
        Tensor,
        Optional[Tensor],
    ]:
        i, f, g, o, b, c, act, act_c = cache.pop()

        B, T, H = i.shape
        dh = zeros((B, H), device=dy.device, dtype=dy.dtype)
        dw_hi = zeros((H, H), device=dy.device, dtype=dy.dtype)
        dw_hf, dw_hg, dw_ho = zeros_like(dw_hi), zeros_like(dw_hi), zeros_like(dw_hi)
        db_hi = None if not b else zeros((H,), device=dy.device, dtype=dy.dtype)
        db_hf = None if not b or not db_hi else zeros_like(db_hi)
        db_hg = None if not b or not db_hi else zeros_like(db_hi)
        db_ho = None if not b or not db_hi else zeros_like(db_hi)
        di_preact, df_preact, dg_preact = empty_like(i), empty_like(f), empty_like(g)
        do_preact, dc = empty_like(o), zeros_like(c)

        for t in range(T - 1, -1, -1):
            # hidden state gradients
            dh += dy[:, t]
            do = act_c[:, t] * dh
            dc[:, t] += act.backward(cache, dh) * o[:, t]

            # cell state gradients
            df = zeros_like(dh) if t < 1 else c[:, t - 1] * dc[:, t]
            if t > 0:
                dc[:, t - 1] += f[:, t] * dc[:, t]
            di = g[:, t] * dc[:, t]
            dg = i[:, t] * dc[:, t]

            # candidate cell state gradients
            dg_preact[:, t] = act.backward(cache, dg)

            # gate gradients
            do_preact[:, t] = SigmoidFn.backward(cache, do)
            df_preact[:, t] = SigmoidFn.backward(cache, df)
            di_preact[:, t] = SigmoidFn.backward(cache, di)

            # hidden projection gradients
            dh_o_t, dw_ho_t, db_ho_t = LinearFn.backward(cache, do_preact[:, t])
            dh_g_t, dw_hg_t, db_hg_t = LinearFn.backward(cache, dg_preact[:, t])
            dh_f_t, dw_hf_t, db_hf_t = LinearFn.backward(cache, df_preact[:, t])
            dh_i_t, dw_hi_t, db_hi_t = LinearFn.backward(cache, di_preact[:, t])

            if t > 0:
                dw_hi += dw_hi_t
                dw_hf += dw_hf_t
                dw_hg += dw_hg_t
                dw_ho += dw_ho_t
            if db_hi_t and db_hf_t and db_hg_t and db_ho_t:
                db_hi += db_hi_t
                db_hf += db_hf_t
                db_hg += db_hg_t
                db_ho += db_ho_t

            dh = dh_i_t + dh_f_t + dh_g_t + dh_o_t

        # input projection gradients
        dx_o, dw_io, db_io = LinearFn.backward(cache, do_preact)
        dx_g, dw_ig, db_ig = LinearFn.backward(cache, dg_preact)
        dx_f, dw_if, db_if = LinearFn.backward(cache, df_preact)
        dx_i, dw_ii, db_ii = LinearFn.backward(cache, di_preact)

        dx = dx_i + dx_f + dx_g + dx_o

        return (
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
        )


def lstm(
    x: Tensor,
    w_ii: Tensor,
    b_ii: Optional[Tensor],
    w_if: Tensor,
    b_if: Optional[Tensor],
    w_ig: Tensor,
    b_ig: Optional[Tensor],
    w_io: Tensor,
    b_io: Optional[Tensor],
    w_hi: Tensor,
    b_hi: Optional[Tensor],
    w_hf: Tensor,
    b_hf: Optional[Tensor],
    w_hg: Tensor,
    b_hg: Optional[Tensor],
    w_ho: Tensor,
    b_ho: Optional[Tensor],
    activation: Literal["relu", "tanh"] = "tanh",
) -> Tensor:
    """Applies the LSTM recurrent function to a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w_ii : Tensor
        Weight tensor for the input projection of the input gate.
    b_ii : Tensor, optional
        Bias tensor for the input projection of the input gate.
    w_if : Tensor
        Weight tensor for the input projection of the forget gate.
    b_if : Tensor, optional
        Bias tensor for the input projection of the forget gate.
    w_ig : Tensor
        Weight tensor for the input projection of the candidate cell state.
    b_ig : Tensor, optional
        Bias tensor for the input projection of the candidate cell state.
    w_io : Tensor
        Weight tensor for the input projection of the output gate.
    b_io : Tensor, optional
        Bias tensor for the input projection of the output gate.
    w_hi : Tensor
        Weight tensor for the hidden projection of the input gate.
    b_hi : Tensor, optional
        Bias tensor for the hidden projection of the input gate.
    w_hf : Tensor
        Weight tensor for the hidden projection of the forget gate.
    b_hf : Tensor, optional
        Bias tensor for the hidden projection of the forget gate.
    w_hg : Tensor
        Weight tensor for the hidden projection of the candidate cell state.
    b_hg : Tensor, optional
        Bias tensor for the hidden projection of the candidate cell state.
    w_ho : Tensor
        Weight tensor for the hidden projection of the output gate.
    b_ho : Tensor, optional
        Bias tensor for the hidden projection of the output gate.
    activation : Literal["relu", "tanh"], optional
        Activation function to use. Defaults to ``tanh``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.LSTM`
    """
    return LSTMFn.forward(
        PseudoCache(),
        x,
        w_ii,
        b_ii,
        w_if,
        b_if,
        w_ig,
        b_ig,
        w_io,
        b_io,
        w_hi,
        b_hi,
        w_hf,
        b_hf,
        w_hg,
        b_hg,
        w_ho,
        b_ho,
        activation,
    )


class GRUFn(Function):
    """Applies the GRU recurrent function to a tensor."""

    @staticmethod
    def forward(
        cache: FunctionCache,
        x: Tensor,
        w_ir: Tensor,
        b_ir: Optional[Tensor],
        w_iz: Tensor,
        b_iz: Optional[Tensor],
        w_in: Tensor,
        b_in: Optional[Tensor],
        w_hr: Tensor,
        b_hr: Optional[Tensor],
        w_hz: Tensor,
        b_hz: Optional[Tensor],
        w_hn: Tensor,
        b_hn: Optional[Tensor],
        activation: str,
    ) -> Tensor:
        if x.ndim != 3:
            raise ShapeError(f"Expected input to be 3D, got {x.ndim}D.")
        if activation not in {"relu", "tanh"}:
            raise ValueError("Activation must be either 'relu' or 'tanh'.")
        act = TanhFn if activation == "tanh" else ReLUFn

        # input projection W_i * x_t + b_i
        x_r = LinearFn.forward(cache, x, w_ir, b_ir)
        x_z = LinearFn.forward(cache, x, w_iz, b_iz)
        x_n = LinearFn.forward(cache, x, w_in, b_in)

        r, z, n = empty_like(x_r), empty_like(x_r), empty_like(x_r)
        h_n, h = empty_like(x_r), zeros_like(x_r)
        for t in range(x.shape[1]):

            # hidden projection W_h * h_t-1 + b_h
            h_r = LinearFn.forward(cache, h[:, t - 1], w_hr, b_hr)
            h_z = LinearFn.forward(cache, h[:, t - 1], w_hz, b_hz)
            h_n[:, t] = LinearFn.forward(cache, h[:, t - 1], w_hn, b_hn)

            # gates
            r[:, t] = SigmoidFn.forward(cache, x_r[:, t] + h_r)  # reset gate
            z[:, t] = SigmoidFn.forward(cache, x_z[:, t] + h_z)  # update gate

            # candidate hidden state n_t = act(x_n + r_t * h_t-1)
            n[:, t] = act.forward(cache, x_n[:, t] + r[:, t] * h_n[:, t])

            # hidden state h_t = (1 - z_t) * n_t + z_t * h_t-1
            h[:, t] = (1 - z[:, t]) * n[:, t] + z[:, t] * h[:, t - 1]

        cache.push(r, z, n, b_iz is not None, h_n, act, h)
        return h

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> tuple[
        Tensor,
        Tensor,
        Optional[Tensor],
        Tensor,
        Optional[Tensor],
        Tensor,
        Optional[Tensor],
        Tensor,
        Optional[Tensor],
        Tensor,
        Optional[Tensor],
        Tensor,
        Optional[Tensor],
    ]:
        r, z, n, b, h_n, act, h = cache.pop()

        # ugly inits :(
        B, T, H = r.shape
        dh = zeros((B, H), device=dy.device, dtype=dy.dtype)
        dw_hr = zeros((H, H), device=dy.device, dtype=dy.dtype)
        dw_hz, dw_hn = zeros_like(dw_hr), zeros_like(dw_hr)
        db_hr = None if not b else zeros((H,), device=dy.device, dtype=dy.dtype)
        db_hz = None if not b or not db_hr else zeros_like(db_hr)
        db_hn = None if not b or not db_hr else zeros_like(db_hr)
        dr_preact, dz_preact, dn_preact = empty_like(r), empty_like(z), empty_like(n)

        for t in range(T - 1, -1, -1):
            # hidden state gradients
            dh += dy[:, t]
            dz = ((0 if t < 1 else h[:, t - 1]) - n[:, t]) * dh
            dn = (1 - z[:, t]) * dh
            dh = z[:, t] * dh

            # candidate hidden state gradients
            dn_preact[:, t] = act.backward(cache, dn)
            dr = h_n[:, t] * dn_preact[:, t]

            # gate gradients
            dz_preact[:, t] = SigmoidFn.backward(cache, dz)
            dr_preact[:, t] = SigmoidFn.backward(cache, dr)

            # hidden projection gradients
            r_dn_preact = r[:, t] * dn_preact[:, t]
            dh_n_t, dw_hn_t, db_hn_t = LinearFn.backward(cache, r_dn_preact)
            dh_z_t, dw_hz_t, db_hz_t = LinearFn.backward(cache, dz_preact[:, t])
            dh_r_t, dw_hr_t, db_hr_t = LinearFn.backward(cache, dr_preact[:, t])

            if t > 0:
                dw_hr += dw_hr_t
                dw_hz += dw_hz_t
                dw_hn += dw_hn_t
            if db_hr_t and db_hz_t and db_hn_t:
                db_hr += db_hr_t
                db_hz += db_hz_t
                db_hn += db_hn_t

            dh += dh_r_t + dh_z_t + dh_n_t

        # input projection gradients
        dx_n, dw_in, db_in = LinearFn.backward(cache, dn_preact)
        dx_z, dw_iz, db_iz = LinearFn.backward(cache, dz_preact)
        dx_r, dw_ir, db_ir = LinearFn.backward(cache, dr_preact)

        dx = dx_r + dx_z + dx_n

        return (
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
        )


def gru(
    x: Tensor,
    w_ir: Tensor,
    b_ir: Optional[Tensor],
    w_iz: Tensor,
    b_iz: Optional[Tensor],
    w_in: Tensor,
    b_in: Optional[Tensor],
    w_hr: Tensor,
    b_hr: Optional[Tensor],
    w_hz: Tensor,
    b_hz: Optional[Tensor],
    w_hn: Tensor,
    b_hn: Optional[Tensor],
    activation: Literal["relu", "tanh"] = "tanh",
) -> Tensor:
    """Applies the GRU recurrent function to a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w_ir : Tensor
        Weight tensor for the input projection of the reset gate.
    b_ir : Tensor, optional
        Bias tensor for the input projection of the reset gate.
    w_iz : Tensor
        Weight tensor for the input projection of the update gate.
    b_iz : Tensor, optional
        Bias tensor for the input projection of the update gate.
    w_in : Tensor
        Weight tensor for the input projection of the candidate hidden state.
    b_in : Tensor, optional
        Bias tensor for the input projection of the candidate hidden state.
    w_hr : Tensor
        Weight tensor for the hidden projection of the reset gate.
    b_hr : Tensor, optional
        Bias tensor for the hidden projection of the reset gate.
    w_hz : Tensor
        Weight tensor for the hidden projection of the update gate.
    b_hz : Tensor, optional
        Bias tensor for the hidden projection of the update gate.
    w_hn : Tensor
        Weight tensor for the hidden projection of the candidate hidden state.
    b_hn : Tensor, optional
        Bias tensor for the hidden projection of the candidate hidden state.
    activation : Literal["relu", "tanh"], optional
        Activation function to use. Defaults to ``tanh``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.GRU`
    """
    return GRUFn.forward(
        PseudoCache(),
        x,
        w_ir,
        b_ir,
        w_iz,
        b_iz,
        w_in,
        b_in,
        w_hr,
        b_hr,
        w_hz,
        b_hz,
        w_hn,
        b_hn,
        activation,
    )
