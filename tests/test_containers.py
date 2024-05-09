"""Containers tests"""

import torch
from compyute.nn.modules.containers import (
    Sequential,
    ParallelConcat,
    ParallelAdd,
)
from compyute.nn.modules.layers import Linear
from tests.test_utils import get_vals_float, get_params, validate


B, Cin, Cout = (10, 20, 30)


def test_sequential_container() -> None:
    """Test for the sequential container."""
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cout, Cin)
    w2_shape = (Cout, Cout)

    # forward
    compyute_x, torch_x = get_vals_float(x_shape)
    compyute_w1, torch_w1 = get_params(w1_shape)
    compyute_w2, torch_w2 = get_params(w2_shape)

    compyute_module = Sequential(
        Linear(Cin, Cout, bias=False),
        Linear(Cout, Cout, bias=False),
    )
    compyute_module.set_training(True)
    compyute_module.modules[0].w = compyute_w1
    compyute_module.modules[1].w = compyute_w2
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Sequential()
    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    torch_module.add_module("lin1", lin1)
    lin2 = torch.nn.Linear(Cout, Cout, bias=False)
    lin2.weight = torch_w2
    torch_module.add_module("lin2", lin2)
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_parallel_concat_container() -> None:
    """Test for the parallel concat container."""
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cout, Cin)
    w2_shape = (Cout, Cin)

    # forward
    compyute_x, torch_x = get_vals_float(x_shape)
    compyute_w1, torch_w1 = get_params(w1_shape)
    compyute_w2, torch_w2 = get_params(w2_shape)

    compyute_module = ParallelConcat(
        Linear(Cin, Cout, bias=False),
        Linear(Cin, Cout, bias=False),
        concat_axis=-1,
    )
    compyute_module.set_training(True)
    compyute_module.modules[0].w = compyute_w1
    compyute_module.modules[1].w = compyute_w2
    compyute_y = compyute_module(compyute_x)

    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    lin2 = torch.nn.Linear(Cin, Cout, bias=False)
    lin2.weight = torch_w2
    torch_parallel_modules = [lin1, lin2]
    torch_y = torch.cat([m(torch_x) for m in torch_parallel_modules], -1)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_parallel_add_container() -> None:
    """Test for the parallel add container."""
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cout, Cin)
    w2_shape = (Cout, Cin)

    # forward
    compyute_x, torch_x = get_vals_float(x_shape)
    compyute_w1, torch_w1 = get_params(w1_shape)
    compyute_w2, torch_w2 = get_params(w2_shape)

    compyute_module = ParallelAdd(
        Linear(Cin, Cout, bias=False),
        Linear(Cin, Cout, bias=False),
    )
    compyute_module.set_training(True)
    compyute_module.modules[0].w = compyute_w1
    compyute_module.modules[1].w = compyute_w2
    compyute_y = compyute_module(compyute_x)

    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    lin2 = torch.nn.Linear(Cin, Cout, bias=False)
    lin2.weight = torch_w2
    torch_y = lin1(torch_x) + lin2(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)
