"""Containers tests"""

import torch

from compyute.nn import Linear, ParallelAdd, ParallelConcat, Sequential
from tests.test_utils import get_random_floats, get_random_params, is_equal

B, Cin, Cout = (10, 20, 30)


def test_sequential_container() -> None:
    """Test for the sequential container."""
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cout, Cin)
    w2_shape = (Cout, Cout)

    # init parameters
    compyute_w1, torch_w1 = get_random_params(w1_shape)
    compyute_w2, torch_w2 = get_random_params(w2_shape)

    # init compyute module
    compyute_lin1 = Linear(Cin, Cout, bias=False, training=True)
    compyute_lin2 = Linear(Cin, Cout, bias=False, training=True)
    compyute_lin1.w = compyute_w1
    compyute_lin2.w = compyute_w2
    compyute_module = Sequential(compyute_lin1, compyute_lin2, training=True)

    # init torch module
    torch_lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    torch_lin2 = torch.nn.Linear(Cout, Cout, bias=False)
    torch_lin1.weight = torch_w1
    torch_lin2.weight = torch_w2
    torch_module = torch.nn.Sequential(torch_lin1, torch_lin2)

    # forward
    compyute_x, torch_x = get_random_floats(x_shape)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    results.append(is_equal(compyute_dx, torch_x.grad))

    assert all(results)


def test_parallel_concat_container() -> None:
    """Test for the parallel concat container."""
    x_shape = (B, Cin)
    w1_shape = (Cout, Cin)
    w2_shape = (Cout, Cin)

    # init parameters
    compyute_w1, torch_w1 = get_random_params(w1_shape)
    compyute_w2, torch_w2 = get_random_params(w2_shape)

    # init compyute module
    compyute_lin1 = Linear(Cin, Cout, bias=False, training=True)
    compyute_lin2 = Linear(Cin, Cout, bias=False, training=True)
    compyute_lin1.w = compyute_w1
    compyute_lin2.w = compyute_w2
    compyute_module = ParallelConcat(compyute_lin1, compyute_lin2, concat_axis=-1, training=True)

    # init torch module
    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin2 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    lin2.weight = torch_w2
    torch_parallel_modules = [lin1, lin2]

    # forward
    compyute_x, torch_x = get_random_floats(x_shape)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch.cat([m(torch_x) for m in torch_parallel_modules], -1)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)


def test_parallel_add_container() -> None:
    """Test for the parallel add container."""
    x_shape = (B, Cin)
    w1_shape = (Cout, Cin)
    w2_shape = (Cout, Cin)

    # init parameters
    compyute_w1, torch_w1 = get_random_params(w1_shape)
    compyute_w2, torch_w2 = get_random_params(w2_shape)

    # init compyute module
    compyute_lin1 = Linear(Cin, Cout, bias=False, training=True)
    compyute_lin2 = Linear(Cin, Cout, bias=False, training=True)
    compyute_lin1.w = compyute_w1
    compyute_lin2.w = compyute_w2
    compyute_module = ParallelAdd(compyute_lin1, compyute_lin2, training=True)

    # init torch
    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin2 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    lin2.weight = torch_w2

    # forward
    compyute_x, torch_x = get_random_floats(x_shape)
    compyute_y = compyute_module(compyute_x)
    torch_y = lin1(torch_x) + lin2(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
