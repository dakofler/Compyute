"""Linear layer tests"""

import torch
import walnut
from tests.test_utils import get_vals, get_params, validate


B, Cin, Cout = (10, 10, 10)


# Linear
def test_linear_cpu() -> None:
    results = []
    shape_x = (B, Cin)
    shape_w = (Cin, Cout)
    shape_b = (Cout,)

    # forward
    walnut_x, torch_x = get_vals(shape_x)
    walnut_w, torch_w = get_params(shape_w, T=True)
    walnut_b, torch_b = get_params(shape_b)

    walnut_module = walnut.nn.layers.Linear(Cin, Cout)
    walnut_module.training = True
    walnut_module.w = walnut_w
    walnut_module.b = walnut_b
    walnut_y = walnut_module(walnut_x)

    torch_module = torch.nn.Linear(Cin, Cout)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False)
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(walnut_dx, torch_x.grad))
    results.append(validate(walnut_module.w.grad, torch_module.weight.grad.T))
    results.append(validate(walnut_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_linear_cuda() -> None:
    results = []
    shape_x = (B, Cin)
    shape_w = (Cin, Cout)
    shape_b = (Cout,)

    # forward
    walnut_x, torch_x = get_vals(shape_x, device="cuda")
    walnut_w, torch_w = get_params(shape_w, T=True, device="cuda")
    walnut_b, torch_b = get_params(shape_b, device="cuda")

    walnut_module = walnut.nn.layers.Linear(Cin, Cout)
    walnut_module.training = True
    walnut_module.to_device("cuda")
    walnut_module.w = walnut_w
    walnut_module.b = walnut_b
    walnut_y = walnut_module(walnut_x)

    torch_module = torch.nn.Linear(Cin, Cout)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False, device="cuda")
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(walnut_dx, torch_x.grad))
    results.append(validate(walnut_module.w.grad, torch_module.weight.grad.T))
    results.append(validate(walnut_module.b.grad, torch_module.bias.grad))

    assert all(results)
