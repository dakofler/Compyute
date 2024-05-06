"""block tests"""

import torch
from compyute.nn.modules.blocks import SkipConnection
from compyute.nn.modules.containers import Sequential
from compyute.nn.modules.layers import Linear, ReLU
from tests.test_utils import get_vals_float, get_params, validate


B, Cin, Cout, X = (10, 20, 30, 40)


# Skip
def test_skip() -> None:
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cout, Cin)
    w2_shape = (Cin, Cout)

    # forward
    compyute_x, torch_x = get_vals_float(x_shape)
    compyute_w1, torch_w1 = get_params(w1_shape)
    compyute_w2, torch_w2 = get_params(w2_shape)

    compyute_module = SkipConnection(
        Sequential(
            [
                Linear(Cin, Cout, bias=False),
                ReLU(),
                Linear(Cout, Cin, bias=False),
            ]
        )
    )
    compyute_module.set_training(True)
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
