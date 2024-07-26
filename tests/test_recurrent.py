"""block tests"""

import torch

from compyute.nn import LSTM, Recurrent, Sequential
from compyute.tensor_functions.combining import concatenate, split
from tests.test_utils import get_random_floats, get_random_params, is_equal

B, Cin, Ch, X = (10, 20, 30, 40)


# Recurrent
def test_recurrent() -> None:
    """Test for the recurrent layer."""
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
    compyute_module = Sequential(
        Recurrent(Cin, Ch, training=True),
        Recurrent(Ch, Ch, training=True, return_sequence=False),
        training=True,
    )
    compyute_module.modules[0].w_i = compyute_w_in_1
    compyute_module.modules[0].b_i = compyute_b_in_1
    compyute_module.modules[0].w_h = compyute_w_h_1
    compyute_module.modules[0].b_h = compyute_b_h_1
    compyute_module.modules[1].w_i = compyute_w_in_2
    compyute_module.modules[1].b_i = compyute_b_in_2
    compyute_module.modules[1].w_h = compyute_w_h_2
    compyute_module.modules[1].b_h = compyute_b_h_2

    # init torch module
    torch_module = torch.nn.RNN(Cin, Ch, batch_first=True, num_layers=2)
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
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)[0][:, -1]  # ouputs tuple of y and hidden_states
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    layers = compyute_module.modules
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(layers[0].w_i.grad, torch_module.weight_ih_l0.grad)
    assert is_equal(layers[0].b_i.grad, torch_module.bias_ih_l0.grad)
    assert is_equal(layers[0].w_h.grad, torch_module.weight_hh_l0.grad)
    assert is_equal(layers[0].b_h.grad, torch_module.bias_hh_l0.grad)
    assert is_equal(layers[1].w_i.grad, torch_module.weight_ih_l1.grad)
    assert is_equal(layers[1].b_i.grad, torch_module.bias_ih_l1.grad)
    assert is_equal(layers[1].w_h.grad, torch_module.weight_hh_l1.grad)
    assert is_equal(layers[1].b_h.grad, torch_module.bias_hh_l1.grad)


# LSTM
def test_lstm() -> None:
    """Test for the lstm layer."""
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
    compyute_w_ii_1, compyute_w_if_1, compyute_w_ig_1, compyute_w_io_1 = split(compyute_w_i_1, 4, 0)
    compyute_b_i_1, torch_b_i_1 = get_random_params(shape_b_i_1)
    compyute_b_ii_1, compyute_b_if_1, compyute_b_ig_1, compyute_b_io_1 = split(compyute_b_i_1, 4, 0)

    # layer 1 hidden
    compyute_w_h_1, torch_w_h_1 = get_random_params(shape_w_h_1)
    compyute_w_hi_1, compyute_w_hf_1, compyute_w_hg_1, compyute_w_ho_1 = split(compyute_w_h_1, 4, 0)
    compyute_b_h_1, torch_b_h_1 = get_random_params(shape_b_h_1)
    compyute_b_hi_1, compyute_b_hf_1, compyute_b_hg_1, compyute_b_ho_1 = split(compyute_b_h_1, 4, 0)

    # layer 2 input
    compyute_w_i_2, torch_w_i_2 = get_random_params(shape_w_i_2)
    compyute_w_ii_2, compyute_w_if_2, compyute_w_ig_2, compyute_w_io_2 = split(compyute_w_i_2, 4, 0)
    compyute_b_i_2, torch_b_i_2 = get_random_params(shape_b_i_2)
    compyute_b_ii_2, compyute_b_if_2, compyute_b_ig_2, compyute_b_io_2 = split(compyute_b_i_2, 4, 0)

    # layer 2 hidden
    compyute_w_h_2, torch_w_h_2 = get_random_params(shape_w_h_2)
    compyute_w_hi_2, compyute_w_hf_2, compyute_w_hg_2, compyute_w_ho_2 = split(compyute_w_h_2, 4, 0)
    compyute_b_h_2, torch_b_h_2 = get_random_params(shape_b_h_2)
    compyute_b_hi_2, compyute_b_hf_2, compyute_b_hg_2, compyute_b_ho_2 = split(compyute_b_h_2, 4, 0)

    # init compyute module
    compyute_module = Sequential(
        LSTM(Cin, Ch, training=True),
        LSTM(Ch, Ch, training=True, return_sequence=False),
        training=True,
    )
    compyute_module.modules[0].w_ii = compyute_w_ii_1
    compyute_module.modules[0].b_ii = compyute_b_ii_1
    compyute_module.modules[0].w_if = compyute_w_if_1
    compyute_module.modules[0].b_if = compyute_b_if_1
    compyute_module.modules[0].w_ig = compyute_w_ig_1
    compyute_module.modules[0].b_ig = compyute_b_ig_1
    compyute_module.modules[0].w_io = compyute_w_io_1
    compyute_module.modules[0].b_io = compyute_b_io_1

    compyute_module.modules[0].w_hi = compyute_w_hi_1
    compyute_module.modules[0].b_hi = compyute_b_hi_1
    compyute_module.modules[0].w_hf = compyute_w_hf_1
    compyute_module.modules[0].b_hf = compyute_b_hf_1
    compyute_module.modules[0].w_hg = compyute_w_hg_1
    compyute_module.modules[0].b_hg = compyute_b_hg_1
    compyute_module.modules[0].w_ho = compyute_w_ho_1
    compyute_module.modules[0].b_ho = compyute_b_ho_1

    compyute_module.modules[1].w_ii = compyute_w_ii_2
    compyute_module.modules[1].b_ii = compyute_b_ii_2
    compyute_module.modules[1].w_if = compyute_w_if_2
    compyute_module.modules[1].b_if = compyute_b_if_2
    compyute_module.modules[1].w_ig = compyute_w_ig_2
    compyute_module.modules[1].b_ig = compyute_b_ig_2
    compyute_module.modules[1].w_io = compyute_w_io_2
    compyute_module.modules[1].b_io = compyute_b_io_2

    compyute_module.modules[1].w_hi = compyute_w_hi_2
    compyute_module.modules[1].b_hi = compyute_b_hi_2
    compyute_module.modules[1].w_hf = compyute_w_hf_2
    compyute_module.modules[1].b_hf = compyute_b_hf_2
    compyute_module.modules[1].w_hg = compyute_w_hg_2
    compyute_module.modules[1].b_hg = compyute_b_hg_2
    compyute_module.modules[1].w_ho = compyute_w_ho_2
    compyute_module.modules[1].b_ho = compyute_b_ho_2

    # init torch module
    torch_module = torch.nn.LSTM(Cin, Ch, batch_first=True, num_layers=2)
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
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)[0][:, -1]  # ouputs tuple of y and hidden_states
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)

    compyute_w_i_1_grad = concatenate(
        [compyute_w_ii_1.grad, compyute_w_if_1.grad, compyute_w_ig_1.grad, compyute_w_io_1.grad], 0
    )
    assert is_equal(compyute_w_i_1_grad, torch_module.weight_ih_l0.grad)

    compyute_b_i_1_grad = concatenate(
        [compyute_b_ii_1.grad, compyute_b_if_1.grad, compyute_b_ig_1.grad, compyute_b_io_1.grad], 0
    )
    assert is_equal(compyute_b_i_1_grad, torch_module.bias_ih_l0.grad)

    compyute_w_h_1_grad = concatenate(
        [compyute_w_hi_1.grad, compyute_w_hf_1.grad, compyute_w_hg_1.grad, compyute_w_ho_1.grad], 0
    )
    assert is_equal(compyute_w_h_1_grad, torch_module.weight_hh_l0.grad)

    compyute_b_h_1_grad = concatenate(
        [compyute_b_hi_1.grad, compyute_b_hf_1.grad, compyute_b_hg_1.grad, compyute_b_ho_1.grad], 0
    )
    assert is_equal(compyute_b_h_1_grad, torch_module.bias_hh_l0.grad)

    compyute_w_i_2_grad = concatenate(
        [compyute_w_ii_2.grad, compyute_w_if_2.grad, compyute_w_ig_2.grad, compyute_w_io_2.grad], 0
    )
    assert is_equal(compyute_w_i_2_grad, torch_module.weight_ih_l1.grad)

    compyute_b_i_2_grad = concatenate(
        [compyute_b_ii_2.grad, compyute_b_if_2.grad, compyute_b_ig_2.grad, compyute_b_io_2.grad], 0
    )
    assert is_equal(compyute_b_i_2_grad, torch_module.bias_ih_l1.grad)

    compyute_w_h_2_grad = concatenate(
        [compyute_w_hi_2.grad, compyute_w_hf_2.grad, compyute_w_hg_2.grad, compyute_w_ho_2.grad], 0
    )
    assert is_equal(compyute_w_h_2_grad, torch_module.weight_hh_l1.grad)

    compyute_b_h_2_grad = concatenate(
        [compyute_b_hi_2.grad, compyute_b_hf_2.grad, compyute_b_hg_2.grad, compyute_b_ho_2.grad], 0
    )
    assert is_equal(compyute_b_h_2_grad, torch_module.bias_hh_l1.grad)
