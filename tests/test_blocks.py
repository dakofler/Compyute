"""block tests"""

import torch
import compyute
from tests.test_utils import get_vals, get_params, validate


B, Cin, Ch, Cout, X = (10, 20, 30, 40, 50)


# Recurrent
def test_recurrent() -> None:
    results = []
    shape_x = (B, X, Cin)
    shape_w_in_1 = (Cin, Ch)
    shape_b_in_1 = (Ch,)
    shape_w_h_1 = (Ch, Ch)
    shape_b_h_1 = (Ch,)
    shape_w_in_2 = (Ch, Ch)
    shape_b_in_2 = (Ch,)
    shape_w_h_2 = (Ch, Ch)
    shape_b_h_2 = (Ch,)

    # forward
    compyute_x, torch_x = get_vals(shape_x)

    compyute_w_in_1, torch_w_in_1 = get_params(shape_w_in_1, T=True)
    compyute_b_in_1, torch_b_in_1 = get_params(shape_b_in_1)

    compyute_w_h_1, torch_w_h_1 = get_params(shape_w_h_1, T=True)
    compyute_b_h_1, torch_b_h_1 = get_params(shape_b_h_1)

    compyute_w_in_2, torch_w_in_2 = get_params(shape_w_in_2, T=True)
    compyute_b_in_2, torch_b_in_2 = get_params(shape_b_in_2)

    compyute_w_h_2, torch_w_h_2 = get_params(shape_w_h_2, T=True)
    compyute_b_h_2, torch_b_h_2 = get_params(shape_b_h_2)

    compyute_module = compyute.nn.blocks.Recurrent(Cin, Ch, num_layers=2)

    compyute_module.training = True

    compyute_module.child_modules[0].w_i = compyute_w_in_1
    compyute_module.child_modules[0].b_i = compyute_b_in_1
    compyute_module.child_modules[0].w_h = compyute_w_h_1
    compyute_module.child_modules[0].b_h = compyute_b_h_1

    compyute_module.child_modules[1].w_i = compyute_w_in_2
    compyute_module.child_modules[1].b_i = compyute_b_in_2
    compyute_module.child_modules[1].w_h = compyute_w_h_2
    compyute_module.child_modules[1].b_h = compyute_b_h_2

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
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)

    # x grad
    results.append(validate(compyute_dx, torch_x.grad))

    # input 1 grads
    results.append(
        validate(
            compyute_module.child_modules[0].w_i.grad,
            torch_module.weight_ih_l0.grad.T,
        )
    )
    results.append(
        validate(
            compyute_module.child_modules[0].b_i.grad, torch_module.bias_ih_l0.grad
        )
    )

    # hidden 1 grads
    results.append(
        validate(
            compyute_module.child_modules[0].w_h.grad, torch_module.weight_hh_l0.grad.T
        )
    )
    results.append(
        validate(
            compyute_module.child_modules[0].b_h.grad, torch_module.bias_hh_l0.grad
        )
    )

    # input 2 grads
    results.append(
        validate(
            compyute_module.child_modules[1].w_i.grad, torch_module.weight_ih_l1.grad.T
        )
    )
    results.append(
        validate(
            compyute_module.child_modules[1].b_i.grad, torch_module.bias_ih_l1.grad
        )
    )

    # hidden 2 grads
    results.append(
        validate(
            compyute_module.child_modules[1].w_h.grad, torch_module.weight_hh_l1.grad.T
        )
    )
    results.append(
        validate(
            compyute_module.child_modules[1].b_h.grad, torch_module.bias_hh_l1.grad
        )
    )

    assert all(results)


# Residual
def test_residual() -> None:
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cin, Cout)
    w2_shape = (Cout, Cin)

    # forward
    compyute_x, torch_x = get_vals(x_shape)
    compyute_w1, torch_w1 = get_params(w1_shape, T=True)
    compyute_w2, torch_w2 = get_params(w2_shape, T=True)

    compyute_module = compyute.nn.blocks.Residual(
        compyute.nn.containers.Sequential(
            [
                compyute.nn.layers.Linear(
                    Cin, Cout, weights=compyute_w1, use_bias=False
                ),
                compyute.nn.layers.ReLU(),
                compyute.nn.layers.Linear(
                    Cout, Cin, weights=compyute_w2, use_bias=False
                ),
            ]
        )
    )
    compyute_module.training = True
    compyute_y = compyute_module(compyute_x)

    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    lin2 = torch.nn.Linear(Cout, Cin, bias=False)
    lin2.weight = torch_w2
    torch_y = torch_x + lin2(torch.nn.functional.relu(lin1(torch_x)))

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)
