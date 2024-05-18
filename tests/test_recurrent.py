"""block tests"""

import torch

from compyute.nn.modules._containers import Sequential
from compyute.nn.modules._layers import LSTM, Recurrent
from tests.test_utils import get_params, get_vals_float, validate

B, Cin, Ch, X = (10, 20, 30, 40)


# Recurrent
def test_recurrent() -> None:
    """Test for the recurrent layer."""
    results = []
    shape_x = (B, X, Cin)
    shape_w_in_1 = (Ch, Cin)
    shape_b_in_1 = (Ch,)
    shape_w_h_1 = (Ch, Ch)
    shape_b_h_1 = (Ch,)
    shape_w_in_2 = (Ch, Ch)
    shape_b_in_2 = (Ch,)
    shape_w_h_2 = (Ch, Ch)
    shape_b_h_2 = (Ch,)

    # forward
    compyute_x, torch_x = get_vals_float(shape_x)

    compyute_w_in_1, torch_w_in_1 = get_params(shape_w_in_1)
    compyute_b_in_1, torch_b_in_1 = get_params(shape_b_in_1)

    compyute_w_h_1, torch_w_h_1 = get_params(shape_w_h_1)
    compyute_b_h_1, torch_b_h_1 = get_params(shape_b_h_1)

    compyute_w_in_2, torch_w_in_2 = get_params(shape_w_in_2)
    compyute_b_in_2, torch_b_in_2 = get_params(shape_b_in_2)

    compyute_w_h_2, torch_w_h_2 = get_params(shape_w_h_2)
    compyute_b_h_2, torch_b_h_2 = get_params(shape_b_h_2)

    compyute_module = Sequential(Recurrent(Cin, Ch), Recurrent(Ch, Ch))
    compyute_module.set_training(True)

    compyute_module.modules[0].w_i = compyute_w_in_1
    compyute_module.modules[0].b_i = compyute_b_in_1
    compyute_module.modules[0].w_h = compyute_w_h_1
    compyute_module.modules[0].b_h = compyute_b_h_1

    compyute_module.modules[1].w_i = compyute_w_in_2
    compyute_module.modules[1].b_i = compyute_b_in_2
    compyute_module.modules[1].w_h = compyute_w_h_2
    compyute_module.modules[1].b_h = compyute_b_h_2

    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.RNN(Cin, Ch, batch_first=True, num_layers=2)

    torch_module.weight_ih_l0 = torch_w_in_1
    torch_module.bias_ih_l0 = torch_b_in_1
    torch_module.weight_hh_l0 = torch_w_h_1
    torch_module.bias_hh_l0 = torch_b_h_1

    torch_module.weight_ih_l1 = torch_w_in_2
    torch_module.bias_ih_l1 = torch_b_in_2
    torch_module.weight_hh_l1 = torch_w_h_2
    torch_module.bias_hh_l1 = torch_b_h_2

    torch_y = torch_module(torch_x)[0]  # ouputs tuple of y and hidden_states

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)

    # x grad
    results.append(validate(compyute_dx, torch_x.grad))

    # input 1 grads
    layers = compyute_module.modules
    results.append(validate(layers[0].w_i.grad, torch_module.weight_ih_l0.grad))
    results.append(validate(layers[0].b_i.grad, torch_module.bias_ih_l0.grad))

    # hidden 1 grads
    results.append(validate(layers[0].w_h.grad, torch_module.weight_hh_l0.grad))
    results.append(validate(layers[0].b_h.grad, torch_module.bias_hh_l0.grad))

    # input 2 grads
    results.append(validate(layers[1].w_i.grad, torch_module.weight_ih_l1.grad))
    results.append(validate(layers[1].b_i.grad, torch_module.bias_ih_l1.grad))

    # hidden 2 grads
    results.append(validate(layers[1].w_h.grad, torch_module.weight_hh_l1.grad))
    results.append(validate(layers[1].b_h.grad, torch_module.bias_hh_l1.grad))

    assert all(results)


# LSTM
def test_lstm() -> None:
    """Test for the lstm layer."""
    results = []
    shape_x = (B, X, Cin)

    shape_w_i_1 = (4 * Ch, Cin)
    shape_b_i_1 = (4 * Ch,)
    shape_w_h_1 = (4 * Ch, Ch)
    shape_b_h_1 = (4 * Ch,)

    shape_w_i_2 = (4 * Ch, Ch)
    shape_b_i_2 = (4 * Ch,)
    shape_w_h_2 = (4 * Ch, Ch)
    shape_b_h_2 = (4 * Ch,)

    # forward
    compyute_x, torch_x = get_vals_float(shape_x)

    compyute_w_i_1, torch_w_i_1 = get_params(shape_w_i_1)
    compyute_b_i_1, torch_b_i_1 = get_params(shape_b_i_1)
    compyute_w_h_1, torch_w_h_1 = get_params(shape_w_h_1)
    compyute_b_h_1, torch_b_h_1 = get_params(shape_b_h_1)

    compyute_w_i_2, torch_w_i_2 = get_params(shape_w_i_2)
    compyute_b_i_2, torch_b_i_2 = get_params(shape_b_i_2)
    compyute_w_h_2, torch_w_h_2 = get_params(shape_w_h_2)
    compyute_b_h_2, torch_b_h_2 = get_params(shape_b_h_2)

    compyute_module = Sequential(LSTM(Cin, Ch), LSTM(Ch, Ch))
    compyute_module.set_training(True)

    compyute_module.modules[0].w_i = compyute_w_i_1
    compyute_module.modules[0].b_i = compyute_b_i_1
    compyute_module.modules[0].w_h = compyute_w_h_1
    compyute_module.modules[0].b_h = compyute_b_h_1

    compyute_module.modules[1].w_i = compyute_w_i_2
    compyute_module.modules[1].b_i = compyute_b_i_2
    compyute_module.modules[1].w_h = compyute_w_h_2
    compyute_module.modules[1].b_h = compyute_b_h_2

    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.LSTM(Cin, Ch, batch_first=True, num_layers=2)

    torch_module.weight_ih_l0 = torch_w_i_1
    torch_module.bias_ih_l0 = torch_b_i_1
    torch_module.weight_hh_l0 = torch_w_h_1
    torch_module.bias_hh_l0 = torch_b_h_1

    torch_module.weight_ih_l1 = torch_w_i_2
    torch_module.bias_ih_l1 = torch_b_i_2
    torch_module.weight_hh_l1 = torch_w_h_2
    torch_module.bias_hh_l1 = torch_b_h_2

    torch_y = torch_module(torch_x)[0]  # ouputs tuple of y and hidden_states

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)

    # x grad
    results.append(validate(compyute_dx, torch_x.grad))

    layers = compyute_module.modules

    # input 1 grads
    results.append(validate(layers[0].w_i.grad, torch_module.weight_ih_l0.grad))
    results.append(validate(layers[0].b_i.grad, torch_module.bias_ih_l0.grad))

    # hidden 1 grads
    results.append(validate(layers[0].w_h.grad, torch_module.weight_hh_l0.grad))
    results.append(validate(layers[0].b_h.grad, torch_module.bias_hh_l0.grad))

    # input 2 grads
    results.append(validate(layers[1].w_i.grad, torch_module.weight_ih_l1.grad))
    results.append(validate(layers[1].b_i.grad, torch_module.bias_ih_l1.grad))

    # hidden 2 grads
    results.append(validate(layers[1].w_h.grad, torch_module.weight_hh_l1.grad))
    results.append(validate(layers[1].b_h.grad, torch_module.bias_hh_l1.grad))

    assert all(results)
