"""Linear layer tests"""

import torch
import compyute
from tests.test_utils import get_vals_float, get_params, validate


B, Bn, Cin, Cout = (10, 20, 30, 40)


def test_linear_2d() -> None:
    results = []
    shape_x = (B, Cin)
    shape_w = (Cin, Cout)
    shape_b = (Cout,)

    # forward
    compyute_x, torch_x = get_vals_float(shape_x)
    compyute_w, torch_w = get_params(shape_w, torch_T=True)
    compyute_b, torch_b = get_params(shape_b)

    compyute_module = compyute.nn.layers.Linear(Cin, Cout)
    compyute_module.training = True
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Linear(Cin, Cout)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad.T))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_linear_nd() -> None:
    results = []
    shape_x = (B, Bn, Cin)
    shape_w = (Cin, Cout)
    shape_b = (Cout,)

    # forward
    compyute_x, torch_x = get_vals_float(shape_x)
    compyute_w, torch_w = get_params(shape_w, torch_T=True)
    compyute_b, torch_b = get_params(shape_b)

    compyute_module = compyute.nn.layers.Linear(Cin, Cout)
    compyute_module.training = True
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Linear(Cin, Cout)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad.T))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)
