"""block tests"""

import torch
from compyute.functional import stack
from compyute.nn.modules.blocks import LSTMBlock, RecurrentBlock, ResidualBlock
from compyute.nn.modules.containers import SequentialContainer
from compyute.nn.modules.layers import Linear, ReLU
from tests.test_utils import get_vals_float, get_params, validate


B, Cin, Ch, Cout, X = (10, 20, 30, 40, 50)


# Recurrent
def test_recurrent() -> None:
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

    compyute_module = RecurrentBlock(Cin, Ch, num_layers=2)

    compyute_module.training = True

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
    compyute_w_ii_1, compyute_w_if_1, compyute_w_ig_1, compyute_w_io_1 = (
        compyute_w_i_1.split([Ch, 2 * Ch, 3 * Ch], axis=0)
    )

    compyute_b_i_1, torch_b_i_1 = get_params(shape_b_i_1)
    compyute_b_ii_1, compyute_b_if_1, compyute_b_ig_1, compyute_b_io_1 = (
        compyute_b_i_1.split([Ch, 2 * Ch, 3 * Ch], axis=0)
    )

    compyute_w_h_1, torch_w_h_1 = get_params(shape_w_h_1)
    compyute_w_hi_1, compyute_w_hf_1, compyute_w_hg_1, compyute_w_ho_1 = (
        compyute_w_h_1.split([Ch, 2 * Ch, 3 * Ch], axis=0)
    )

    compyute_b_h_1, torch_b_h_1 = get_params(shape_b_h_1)
    compyute_b_hi_1, compyute_b_hf_1, compyute_b_hg_1, compyute_b_ho_1 = (
        compyute_b_h_1.split([Ch, 2 * Ch, 3 * Ch], axis=0)
    )

    compyute_w_i_2, torch_w_i_2 = get_params(shape_w_i_2)
    compyute_w_ii_2, compyute_w_if_2, compyute_w_ig_2, compyute_w_io_2 = (
        compyute_w_i_2.split([Ch, 2 * Ch, 3 * Ch], axis=0)
    )

    compyute_b_i_2, torch_b_i_2 = get_params(shape_b_i_2)
    compyute_b_ii_2, compyute_b_if_2, compyute_b_ig_2, compyute_b_io_2 = (
        compyute_b_i_2.split([Ch, 2 * Ch, 3 * Ch], axis=0)
    )

    compyute_w_h_2, torch_w_h_2 = get_params(shape_w_h_2)
    compyute_w_hi_2, compyute_w_hf_2, compyute_w_hg_2, compyute_w_ho_2 = (
        compyute_w_h_2.split([Ch, 2 * Ch, 3 * Ch], axis=0)
    )

    compyute_b_h_2, torch_b_h_2 = get_params(shape_b_h_2)
    compyute_b_hi_2, compyute_b_hf_2, compyute_b_hg_2, compyute_b_ho_2 = (
        compyute_b_h_2.split([Ch, 2 * Ch, 3 * Ch], axis=0)
    )

    compyute_module = LSTMBlock(Cin, Ch, num_layers=2)

    compyute_module.training = True

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
    w_i_1_grads = stack(
        [
            layers[0].w_ii.grad,
            layers[0].w_if.grad,
            layers[0].w_ig.grad,
            layers[0].w_io.grad,
        ]
    )
    results.append(validate(w_i_1_grads, torch_module.weight_ih_l0.grad))
    b_i_1_grads = stack(
        [
            layers[0].b_ii.grad,
            layers[0].b_if.grad,
            layers[0].b_ig.grad,
            layers[0].b_io.grad,
        ]
    )
    results.append(validate(b_i_1_grads, torch_module.bias_ih_l0.grad))

    # hidden 1 grads
    w_h_1_grads = stack(
        [
            layers[0].w_hi.grad,
            layers[0].w_hf.grad,
            layers[0].w_hg.grad,
            layers[0].w_ho.grad,
        ]
    )
    results.append(validate(w_h_1_grads, torch_module.weight_hh_l0.grad))
    b_h_1_grads = stack(
        [
            layers[0].b_hi.grad,
            layers[0].b_hf.grad,
            layers[0].b_hg.grad,
            layers[0].b_ho.grad,
        ]
    )
    results.append(validate(b_h_1_grads, torch_module.bias_hh_l0.grad))

    # input 2 grads
    w_i_2_grads = stack(
        [
            layers[1].w_ii.grad,
            layers[1].w_if.grad,
            layers[1].w_ig.grad,
            layers[1].w_io.grad,
        ]
    )
    results.append(validate(w_i_2_grads, torch_module.weight_ih_l1.grad))
    b_i_2_grads = stack(
        [
            layers[1].b_ii.grad,
            layers[1].b_if.grad,
            layers[1].b_ig.grad,
            layers[1].b_io.grad,
        ]
    )
    results.append(validate(b_i_2_grads, torch_module.bias_ih_l1.grad))

    # hidden 2 grads
    w_h_2_grads = stack(
        [
            layers[1].w_hi.grad,
            layers[1].w_hf.grad,
            layers[1].w_hg.grad,
            layers[1].w_ho.grad,
        ]
    )
    results.append(validate(w_h_2_grads, torch_module.weight_hh_l1.grad))
    b_h_2_grads = stack(
        [
            layers[1].b_hi.grad,
            layers[1].b_hf.grad,
            layers[1].b_hg.grad,
            layers[1].b_ho.grad,
        ]
    )
    results.append(validate(b_h_2_grads, torch_module.bias_hh_l1.grad))

    assert all(results)


# Residual
def test_residual() -> None:
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cout, Cin)
    w2_shape = (Cin, Cout)

    # forward
    compyute_x, torch_x = get_vals_float(x_shape)
    compyute_w1, torch_w1 = get_params(w1_shape)
    compyute_w2, torch_w2 = get_params(w2_shape)

    compyute_module = ResidualBlock(
        SequentialContainer(
            [
                Linear(Cin, Cout, bias=False),
                ReLU(),
                Linear(Cout, Cin, bias=False),
            ]
        )
    )
    compyute_module.training = True
    compyute_module.modules[0].modules[0].w = compyute_w1
    compyute_module.modules[0].modules[2].w = compyute_w2
    compyute_y = compyute_module(compyute_x)

    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    lin2 = torch.nn.Linear(Cout, Cin, bias=False)
    lin2.weight = torch_w2
    torch_y = torch_x + lin2(torch.nn.functional.relu(lin1(torch_x)))

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)
