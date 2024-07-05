"""block tests"""

import torch

from compyute.nn import LSTM, Recurrent, Sequential
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
        Recurrent(Cin, Ch, training=True), Recurrent(Ch, Ch, training=True), training=True
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
    torch_y = torch_module(torch_x)[0]  # ouputs tuple of y and hidden_states
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
    compyute_w_i_1, torch_w_i_1 = get_random_params(shape_w_i_1)
    compyute_b_i_1, torch_b_i_1 = get_random_params(shape_b_i_1)
    compyute_w_h_1, torch_w_h_1 = get_random_params(shape_w_h_1)
    compyute_b_h_1, torch_b_h_1 = get_random_params(shape_b_h_1)
    compyute_w_i_2, torch_w_i_2 = get_random_params(shape_w_i_2)
    compyute_b_i_2, torch_b_i_2 = get_random_params(shape_b_i_2)
    compyute_w_h_2, torch_w_h_2 = get_random_params(shape_w_h_2)
    compyute_b_h_2, torch_b_h_2 = get_random_params(shape_b_h_2)

    compyute_module = Sequential(
        LSTM(Cin, Ch, training=True), LSTM(Ch, Ch, training=True), training=True
    )
    compyute_module.modules[0].w_i = compyute_w_i_1
    compyute_module.modules[0].b_i = compyute_b_i_1
    compyute_module.modules[0].w_h = compyute_w_h_1
    compyute_module.modules[0].b_h = compyute_b_h_1
    compyute_module.modules[1].w_i = compyute_w_i_2
    compyute_module.modules[1].b_i = compyute_b_i_2
    compyute_module.modules[1].w_h = compyute_w_h_2
    compyute_module.modules[1].b_h = compyute_b_h_2

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
    torch_y = torch_module(torch_x)[0]  # ouputs tuple of y and hidden_states
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
