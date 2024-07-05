"""block tests"""

import torch

from src.compyute.nn import Linear, ReLU, Sequential, SkipConnection
from tests.test_utils import get_random_floats, get_random_params, is_equal

B, Cin, Cout, X = (10, 20, 30, 40)


def test_skip() -> None:
    """Test for the skip connection."""
    x_shape = (B, Cin)
    w1_shape = (Cout, Cin)
    w2_shape = (Cin, Cout)

    # init parameters
    compyute_w1, torch_w1 = get_random_params(w1_shape)
    compyute_w2, torch_w2 = get_random_params(w2_shape)

    # init compyute module
    compyute_lin1 = Linear(Cin, Cout, bias=False, training=True)
    compyute_lin2 = Linear(Cin, Cout, bias=False, training=True)
    compyute_lin1.w = compyute_w1
    compyute_lin2.w = compyute_w2
    compyute_module = SkipConnection(
        Sequential(compyute_lin1, ReLU(training=True), compyute_lin2, training=True), training=True
    )

    # init torch module
    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin2 = torch.nn.Linear(Cout, Cin, bias=False)
    lin1.weight = torch_w1
    lin2.weight = torch_w2

    # forward
    compyute_x, torch_x = get_random_floats(x_shape)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_x + lin2(torch.nn.functional.relu(lin1(torch_x)))
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
