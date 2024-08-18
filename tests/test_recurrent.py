"""Recurrent module tests"""

import pytest
import torch

from compyute.nn import GRU, LSTM, Parameter, Recurrent, Sequential
from compyute.tensor_ops.creating import concat, split
from tests.test_utils import get_random_floats, get_random_params, is_equal

shape_testdata = [(8, 16, 32, 64), (16, 32, 64, 128)]
act_testdata = ["tanh", "relu"]


@pytest.mark.parametrize("shape", shape_testdata)
@pytest.mark.parametrize("act", act_testdata)
def test_recurrent(shape, act) -> None:
    """Test for the recurrent layer."""

    B, X, Cin, Ch = shape
    shape_x = (B, X, Cin)
    shape_w_in_1 = (Ch, Cin)
    shape_b_in_1 = (Ch,)
    shape_w_h_1 = (Ch, Ch)
    shape_b_h_1 = (Ch,)
    shape_w_in_2 = (Ch, Ch)
    shape_b_in_2 = (Ch,)
    shape_w_h_2 = (Ch, Ch)
    shape_b_h_2 = (Ch,)

    # init parameters
    compyute_w_in_1, torch_w_in_1 = get_random_params(shape_w_in_1)
    compyute_b_in_1, torch_b_in_1 = get_random_params(shape_b_in_1)
    compyute_w_h_1, torch_w_h_1 = get_random_params(shape_w_h_1)
    compyute_b_h_1, torch_b_h_1 = get_random_params(shape_b_h_1)
    compyute_w_in_2, torch_w_in_2 = get_random_params(shape_w_in_2)
    compyute_b_in_2, torch_b_in_2 = get_random_params(shape_b_in_2)
    compyute_w_h_2, torch_w_h_2 = get_random_params(shape_w_h_2)
    compyute_b_h_2, torch_b_h_2 = get_random_params(shape_b_h_2)

    # init compyute module
    compyute_recurrent1 = Recurrent(Cin, Ch, activation=act)
    compyute_recurrent2 = Recurrent(Ch, Ch, activation=act, return_sequence=False)
    compyute_recurrent1.w_i = compyute_w_in_1
    compyute_recurrent1.b_i = compyute_b_in_1
    compyute_recurrent1.w_h = compyute_w_h_1
    compyute_recurrent1.b_h = compyute_b_h_1
    compyute_recurrent2.w_i = compyute_w_in_2
    compyute_recurrent2.b_i = compyute_b_in_2
    compyute_recurrent2.w_h = compyute_w_h_2
    compyute_recurrent2.b_h = compyute_b_h_2
    compyute_module = Sequential(compyute_recurrent1, compyute_recurrent2)

    # init torch module
    torch_module = torch.nn.RNN(Cin, Ch, 2, act, batch_first=True)
    torch_module.weight_ih_l0 = torch_w_in_1
    torch_module.bias_ih_l0 = torch_b_in_1
    torch_module.weight_hh_l0 = torch_w_h_1
    torch_module.bias_hh_l0 = torch_b_h_1
    torch_module.weight_ih_l1 = torch_w_in_2
    torch_module.bias_ih_l1 = torch_b_in_2
    torch_module.weight_hh_l1 = torch_w_h_2
    torch_module.bias_hh_l1 = torch_b_h_2

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    with compyute_module.train():
        compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)[0][:, -1]  # outputs tuple of y and hidden_states
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    with compyute_module.train():
        compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)

    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_w_in_1.grad, torch_module.weight_ih_l0.grad)
    assert is_equal(compyute_b_in_1.grad, torch_module.bias_ih_l0.grad)
    assert is_equal(compyute_w_h_1.grad, torch_module.weight_hh_l0.grad)
    assert is_equal(compyute_b_h_1.grad, torch_module.bias_hh_l0.grad)
    assert is_equal(compyute_w_in_2.grad, torch_module.weight_ih_l1.grad)
    assert is_equal(compyute_b_in_2.grad, torch_module.bias_ih_l1.grad)
    assert is_equal(compyute_w_h_2.grad, torch_module.weight_hh_l1.grad)
    assert is_equal(compyute_b_h_2.grad, torch_module.bias_hh_l1.grad)


@pytest.mark.parametrize("shape", shape_testdata)
def test_lstm(shape) -> None:
    """Test for the lstm layer."""

    B, X, Cin, Ch = shape
    shape_x = (B, X, Cin)
    shape_w_i_1 = (4 * Ch, Cin)
    shape_b_i_1 = (4 * Ch,)
    shape_w_h_1 = (4 * Ch, Ch)
    shape_b_h_1 = (4 * Ch,)
    shape_w_i_2 = (4 * Ch, Ch)
    shape_b_i_2 = (4 * Ch,)
    shape_w_h_2 = (4 * Ch, Ch)
    shape_b_h_2 = (4 * Ch,)

    # init parameters
    # layer 1 input
    compyute_w_i_1, torch_w_i_1 = get_random_params(shape_w_i_1)
    compyute_w_ii_1, compyute_w_if_1, compyute_w_ig_1, compyute_w_io_1 = split(
        compyute_w_i_1, 4, 0
    )
    compyute_b_i_1, torch_b_i_1 = get_random_params(shape_b_i_1)
    compyute_b_ii_1, compyute_b_if_1, compyute_b_ig_1, compyute_b_io_1 = split(
        compyute_b_i_1, 4, 0
    )

    # layer 1 hidden
    compyute_w_h_1, torch_w_h_1 = get_random_params(shape_w_h_1)
    compyute_w_hi_1, compyute_w_hf_1, compyute_w_hg_1, compyute_w_ho_1 = split(
        compyute_w_h_1, 4, 0
    )
    compyute_b_h_1, torch_b_h_1 = get_random_params(shape_b_h_1)
    compyute_b_hi_1, compyute_b_hf_1, compyute_b_hg_1, compyute_b_ho_1 = split(
        compyute_b_h_1, 4, 0
    )

    # layer 2 input
    compyute_w_i_2, torch_w_i_2 = get_random_params(shape_w_i_2)
    compyute_w_ii_2, compyute_w_if_2, compyute_w_ig_2, compyute_w_io_2 = split(
        compyute_w_i_2, 4, 0
    )
    compyute_b_i_2, torch_b_i_2 = get_random_params(shape_b_i_2)
    compyute_b_ii_2, compyute_b_if_2, compyute_b_ig_2, compyute_b_io_2 = split(
        compyute_b_i_2, 4, 0
    )

    # layer 2 hidden
    compyute_w_h_2, torch_w_h_2 = get_random_params(shape_w_h_2)
    compyute_w_hi_2, compyute_w_hf_2, compyute_w_hg_2, compyute_w_ho_2 = split(
        compyute_w_h_2, 4, 0
    )
    compyute_b_h_2, torch_b_h_2 = get_random_params(shape_b_h_2)
    compyute_b_hi_2, compyute_b_hf_2, compyute_b_hg_2, compyute_b_ho_2 = split(
        compyute_b_h_2, 4, 0
    )

    # init compyute module
    lstm1 = LSTM(Cin, Ch)
    lstm1.w_ii = Parameter(compyute_w_ii_1)
    lstm1.b_ii = Parameter(compyute_b_ii_1)
    lstm1.w_if = Parameter(compyute_w_if_1)
    lstm1.b_if = Parameter(compyute_b_if_1)
    lstm1.w_ig = Parameter(compyute_w_ig_1)
    lstm1.b_ig = Parameter(compyute_b_ig_1)
    lstm1.w_io = Parameter(compyute_w_io_1)
    lstm1.b_io = Parameter(compyute_b_io_1)

    lstm1.w_hi = Parameter(compyute_w_hi_1)
    lstm1.b_hi = Parameter(compyute_b_hi_1)
    lstm1.w_hf = Parameter(compyute_w_hf_1)
    lstm1.b_hf = Parameter(compyute_b_hf_1)
    lstm1.w_hg = Parameter(compyute_w_hg_1)
    lstm1.b_hg = Parameter(compyute_b_hg_1)
    lstm1.w_ho = Parameter(compyute_w_ho_1)
    lstm1.b_ho = Parameter(compyute_b_ho_1)

    lstm2 = LSTM(Ch, Ch, return_sequence=False)
    lstm2.w_ii = Parameter(compyute_w_ii_2)
    lstm2.b_ii = Parameter(compyute_b_ii_2)
    lstm2.w_if = Parameter(compyute_w_if_2)
    lstm2.b_if = Parameter(compyute_b_if_2)
    lstm2.w_ig = Parameter(compyute_w_ig_2)
    lstm2.b_ig = Parameter(compyute_b_ig_2)
    lstm2.w_io = Parameter(compyute_w_io_2)
    lstm2.b_io = Parameter(compyute_b_io_2)

    lstm2.w_hi = Parameter(compyute_w_hi_2)
    lstm2.b_hi = Parameter(compyute_b_hi_2)
    lstm2.w_hf = Parameter(compyute_w_hf_2)
    lstm2.b_hf = Parameter(compyute_b_hf_2)
    lstm2.w_hg = Parameter(compyute_w_hg_2)
    lstm2.b_hg = Parameter(compyute_b_hg_2)
    lstm2.w_ho = Parameter(compyute_w_ho_2)
    lstm2.b_ho = Parameter(compyute_b_ho_2)

    compyute_module = Sequential(lstm1, lstm2)

    # init torch module
    torch_module = torch.nn.LSTM(Cin, Ch, 2, batch_first=True)
    torch_module.weight_ih_l0 = torch_w_i_1
    torch_module.bias_ih_l0 = torch_b_i_1
    torch_module.weight_hh_l0 = torch_w_h_1
    torch_module.bias_hh_l0 = torch_b_h_1
    torch_module.weight_ih_l1 = torch_w_i_2
    torch_module.bias_ih_l1 = torch_b_i_2
    torch_module.weight_hh_l1 = torch_w_h_2
    torch_module.bias_hh_l1 = torch_b_h_2

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    with compyute_module.train():
        compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)[0][:, -1]  # outputs tuple of y and hidden_states
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    with compyute_module.train():
        compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)

    compyute_w_i_1_grad = concat(
        [lstm1.w_ii.grad, lstm1.w_if.grad, lstm1.w_ig.grad, lstm1.w_io.grad], 0
    )
    assert is_equal(compyute_w_i_1_grad, torch_module.weight_ih_l0.grad)

    compyute_b_i_1_grad = concat(
        [lstm1.b_ii.grad, lstm1.b_if.grad, lstm1.b_ig.grad, lstm1.b_io.grad], 0
    )
    assert is_equal(compyute_b_i_1_grad, torch_module.bias_ih_l0.grad)

    compyute_w_h_1_grad = concat(
        [lstm1.w_hi.grad, lstm1.w_hf.grad, lstm1.w_hg.grad, lstm1.w_ho.grad], 0
    )
    assert is_equal(compyute_w_h_1_grad, torch_module.weight_hh_l0.grad)

    compyute_b_h_1_grad = concat(
        [lstm1.b_hi.grad, lstm1.b_hf.grad, lstm1.b_hg.grad, lstm1.b_ho.grad], 0
    )
    assert is_equal(compyute_b_h_1_grad, torch_module.bias_hh_l0.grad)

    compyute_w_i_2_grad = concat(
        [lstm2.w_ii.grad, lstm2.w_if.grad, lstm2.w_ig.grad, lstm2.w_io.grad], 0
    )
    assert is_equal(compyute_w_i_2_grad, torch_module.weight_ih_l1.grad)

    compyute_b_i_2_grad = concat(
        [lstm2.b_ii.grad, lstm2.b_if.grad, lstm2.b_ig.grad, lstm2.b_io.grad], 0
    )
    assert is_equal(compyute_b_i_2_grad, torch_module.bias_ih_l1.grad)

    compyute_w_h_2_grad = concat(
        [lstm2.w_hi.grad, lstm2.w_hf.grad, lstm2.w_hg.grad, lstm2.w_ho.grad], 0
    )
    assert is_equal(compyute_w_h_2_grad, torch_module.weight_hh_l1.grad)

    compyute_b_h_2_grad = concat(
        [lstm2.b_hi.grad, lstm2.b_hf.grad, lstm2.b_hg.grad, lstm2.b_ho.grad], 0
    )
    assert is_equal(compyute_b_h_2_grad, torch_module.bias_hh_l1.grad)


@pytest.mark.parametrize("shape", shape_testdata)
def test_gru(shape) -> None:
    """Test for the gru layer."""

    B, X, Cin, Ch = shape
    shape_x = (B, X, Cin)
    shape_w_i_1 = (3 * Ch, Cin)
    shape_b_i_1 = (3 * Ch,)
    shape_w_h_1 = (3 * Ch, Ch)
    shape_b_h_1 = (3 * Ch,)
    shape_w_i_2 = (3 * Ch, Ch)
    shape_b_i_2 = (3 * Ch,)
    shape_w_h_2 = (3 * Ch, Ch)
    shape_b_h_2 = (3 * Ch,)

    # init parameters
    # layer 1 input
    compyute_w_i_1, torch_w_i_1 = get_random_params(shape_w_i_1)
    compyute_w_ir_1, compyute_w_iz_1, compyute_w_in_1 = split(compyute_w_i_1, 3, 0)
    compyute_b_i_1, torch_b_i_1 = get_random_params(shape_b_i_1)
    compyute_b_ir_1, compyute_b_iz_1, compyute_b_in_1 = split(compyute_b_i_1, 3, 0)

    # layer 1 hidden
    compyute_w_h_1, torch_w_h_1 = get_random_params(shape_w_h_1)
    compyute_w_hr_1, compyute_w_hz_1, compyute_w_hn_1 = split(compyute_w_h_1, 3, 0)
    compyute_b_h_1, torch_b_h_1 = get_random_params(shape_b_h_1)
    compyute_b_hr_1, compyute_b_hz_1, compyute_b_hn_1 = split(compyute_b_h_1, 3, 0)

    # layer 2 input
    compyute_w_i_2, torch_w_i_2 = get_random_params(shape_w_i_2)
    compyute_w_ir_2, compyute_w_iz_2, compyute_w_in_2 = split(compyute_w_i_2, 3, 0)
    compyute_b_i_2, torch_b_i_2 = get_random_params(shape_b_i_2)
    compyute_b_ir_2, compyute_b_iz_2, compyute_b_in_2 = split(compyute_b_i_2, 3, 0)

    # layer 2 hidden
    compyute_w_h_2, torch_w_h_2 = get_random_params(shape_w_h_2)
    compyute_w_hr_2, compyute_w_hz_2, compyute_w_hn_2 = split(compyute_w_h_2, 3, 0)
    compyute_b_h_2, torch_b_h_2 = get_random_params(shape_b_h_2)
    compyute_b_hr_2, compyute_b_hz_2, compyute_b_hn_2 = split(compyute_b_h_2, 3, 0)

    # init compyute module
    gru1 = GRU(Cin, Ch)
    gru1.w_ir = Parameter(compyute_w_ir_1)
    gru1.b_ir = Parameter(compyute_b_ir_1)
    gru1.w_iz = Parameter(compyute_w_iz_1)
    gru1.b_iz = Parameter(compyute_b_iz_1)
    gru1.w_in = Parameter(compyute_w_in_1)
    gru1.b_in = Parameter(compyute_b_in_1)
    gru1.w_hr = Parameter(compyute_w_hr_1)
    gru1.b_hr = Parameter(compyute_b_hr_1)
    gru1.w_hz = Parameter(compyute_w_hz_1)
    gru1.b_hz = Parameter(compyute_b_hz_1)
    gru1.w_hn = Parameter(compyute_w_hn_1)
    gru1.b_hn = Parameter(compyute_b_hn_1)

    gru2 = GRU(Ch, Ch, return_sequence=False)
    gru2.w_ir = Parameter(compyute_w_ir_2)
    gru2.b_ir = Parameter(compyute_b_ir_2)
    gru2.w_iz = Parameter(compyute_w_iz_2)
    gru2.b_iz = Parameter(compyute_b_iz_2)
    gru2.w_in = Parameter(compyute_w_in_2)
    gru2.b_in = Parameter(compyute_b_in_2)
    gru2.w_hr = Parameter(compyute_w_hr_2)
    gru2.b_hr = Parameter(compyute_b_hr_2)
    gru2.w_hz = Parameter(compyute_w_hz_2)
    gru2.b_hz = Parameter(compyute_b_hz_2)
    gru2.w_hn = Parameter(compyute_w_hn_2)
    gru2.b_hn = Parameter(compyute_b_hn_2)

    compyute_module = Sequential(gru1, gru2)

    # init torch module
    torch_module = torch.nn.GRU(Cin, Ch, num_layers=2, batch_first=True)
    torch_module.weight_ih_l0 = torch_w_i_1
    torch_module.bias_ih_l0 = torch_b_i_1
    torch_module.weight_hh_l0 = torch_w_h_1
    torch_module.bias_hh_l0 = torch_b_h_1
    torch_module.weight_ih_l1 = torch_w_i_2
    torch_module.bias_ih_l1 = torch_b_i_2
    torch_module.weight_hh_l1 = torch_w_h_2
    torch_module.bias_hh_l1 = torch_b_h_2

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    with compyute_module.train():
        compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)[0][:, -1]  # outputs tuple of y and hidden_states
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    with compyute_module.train():
        compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)

    compyute_w_i_1_grad = concat([gru1.w_ir.grad, gru1.w_iz.grad, gru1.w_in.grad], 0)
    assert is_equal(compyute_w_i_1_grad, torch_module.weight_ih_l0.grad)

    compyute_b_i_1_grad = concat([gru1.b_ir.grad, gru1.b_iz.grad, gru1.b_in.grad], 0)
    assert is_equal(compyute_b_i_1_grad, torch_module.bias_ih_l0.grad)

    compyute_w_h_1_grad = concat([gru1.w_hr.grad, gru1.w_hz.grad, gru1.w_hn.grad], 0)
    assert is_equal(compyute_w_h_1_grad, torch_module.weight_hh_l0.grad)

    compyute_b_h_1_grad = concat([gru1.b_hr.grad, gru1.b_hz.grad, gru1.b_hn.grad], 0)
    assert is_equal(compyute_b_h_1_grad, torch_module.bias_hh_l0.grad)

    compyute_w_i_2_grad = concat([gru2.w_ir.grad, gru2.w_iz.grad, gru2.w_in.grad], 0)
    assert is_equal(compyute_w_i_2_grad, torch_module.weight_ih_l1.grad)

    compyute_b_i_2_grad = concat([gru2.b_ir.grad, gru2.b_iz.grad, gru2.b_in.grad], 0)
    assert is_equal(compyute_b_i_2_grad, torch_module.bias_ih_l1.grad)

    compyute_w_h_2_grad = concat([gru2.w_hr.grad, gru2.w_hz.grad, gru2.w_hn.grad], 0)
    assert is_equal(compyute_w_h_2_grad, torch_module.weight_hh_l1.grad)

    compyute_b_h_2_grad = concat([gru2.b_hr.grad, gru2.b_hz.grad, gru2.b_hn.grad], 0)
    assert is_equal(compyute_b_h_2_grad, torch_module.bias_hh_l1.grad)
