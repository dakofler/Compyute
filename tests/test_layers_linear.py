"""Linear module tests"""

import torch

from compyute.nn import Linear
from tests.test_utils import get_random_floats, get_random_params, is_equal

B, Bn, Cin, Cout = (10, 20, 30, 40)


def test_linear_2d() -> None:
    """Test for the linear layer using 2d inputs."""
    shape_x = (B, Cin)
    shape_w = (Cout, Cin)
    shape_b = (Cout,)

    # init parameters
    compyute_w, torch_w = get_random_params(shape_w)
    compyute_b, torch_b = get_random_params(shape_b)

    # init compyute module
    compyute_module = Linear(Cin, Cout, training=True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b

    # init torch module
    torch_module = torch.nn.Linear(Cin, Cout)
    torch_module.weight = torch_w
    torch_module.bias = torch_b

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)


def test_linear_nd() -> None:
    """Test for the linear layer using nd inputs."""
    shape_x = (B, Bn, Cin)
    shape_w = (Cout, Cin)
    shape_b = (Cout,)

    # init parameters
    compyute_w, torch_w = get_random_params(shape_w)
    compyute_b, torch_b = get_random_params(shape_b)

    # init compyute module
    compyute_module = Linear(Cin, Cout, training=True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b

    # init torch module
    torch_module = torch.nn.Linear(Cin, Cout)
    torch_module.weight = torch_w
    torch_module.bias = torch_b

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)

    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)
