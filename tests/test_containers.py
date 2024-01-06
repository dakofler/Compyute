"""Containers tests"""

import torch
import walnut
from tests.test_utils import get_vals, get_params, validate


B, Cin, Cout = (10, 10, 10)


# Sequential container
def test_sequential_container_cpu() -> None:
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cin, Cout)
    w2_shape = (Cout, Cout)

    # forward
    walnut_x, torch_x = get_vals(x_shape)
    walnut_w1, torch_w1 = get_params(w1_shape, T=True)
    walnut_w2, torch_w2 = get_params(w2_shape, T=True)

    walnut_module = walnut.nn.containers.SequentialContainer(
        [
            walnut.nn.layers.Linear(Cin, Cout, weights=walnut_w1, use_bias=False),
            walnut.nn.layers.Linear(Cout, Cout, weights=walnut_w2, use_bias=False),
        ]
    )
    walnut_module.training = True
    walnut_y = walnut_module(walnut_x)

    torch_module = torch.nn.Sequential()
    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    torch_module.add_module("lin1", lin1)
    lin2 = torch.nn.Linear(Cout, Cout, bias=False)
    lin2.weight = torch_w2
    torch_module.add_module("lin2", lin2)
    torch_y = torch_module(torch_x)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False)
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))

    assert all(results)


# Sequential container
def test_sequential_container_cuda() -> None:
    if not walnut.cuda.is_available():
        pass
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cin, Cout)
    w2_shape = (Cout, Cout)

    # forward
    walnut_x, torch_x = get_vals(x_shape, device="cuda")
    walnut_w1, torch_w1 = get_params(w1_shape, T=True, device="cuda")
    walnut_w2, torch_w2 = get_params(w2_shape, T=True, device="cuda")

    walnut_module = walnut.nn.containers.SequentialContainer(
        [
            walnut.nn.layers.Linear(Cin, Cout, weights=walnut_w1, use_bias=False),
            walnut.nn.layers.Linear(Cout, Cout, weights=walnut_w2, use_bias=False),
        ]
    )
    walnut_module.training = True
    walnut_module.to_device("cuda")
    walnut_y = walnut_module(walnut_x)

    torch_module = torch.nn.Sequential()
    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    torch_module.add_module("lin1", lin1)
    lin2 = torch.nn.Linear(Cout, Cout, bias=False)
    lin2.weight = torch_w2
    torch_module.add_module("lin2", lin2)
    torch_y = torch_module(torch_x)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False, device="cuda")
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))

    assert all(results)


# Parallel container
def test_parallel_container_cpu() -> None:
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cin, Cout)
    w2_shape = (Cout, Cout)

    # forward
    walnut_x, torch_x = get_vals(x_shape)
    walnut_w1, torch_w1 = get_params(w1_shape, T=True)
    walnut_w2, torch_w2 = get_params(w2_shape, T=True)

    walnut_module = walnut.nn.containers.ParallelContainer(
        [
            walnut.nn.layers.Linear(Cin, Cout, weights=walnut_w1, use_bias=False),
            walnut.nn.layers.Linear(Cout, Cout, weights=walnut_w2, use_bias=False),
        ],
        -1,
    )
    walnut_module.training = True
    walnut_y = walnut_module(walnut_x)

    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    lin2 = torch.nn.Linear(Cout, Cout, bias=False)
    lin2.weight = torch_w2
    torch_parallel_modules = [lin1, lin2]
    torch_y = torch.cat([m(torch_x) for m in torch_parallel_modules], -1)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False)
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))

    assert all(results)


def test_parallel_container_cuda() -> None:
    if not walnut.cuda.is_available():
        pass
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cin, Cout)
    w2_shape = (Cout, Cout)

    # forward
    walnut_x, torch_x = get_vals(x_shape, device="cuda")
    walnut_w1, torch_w1 = get_params(w1_shape, T=True, device="cuda")
    walnut_w2, torch_w2 = get_params(w2_shape, T=True, device="cuda")

    walnut_module = walnut.nn.containers.ParallelContainer(
        [
            walnut.nn.layers.Linear(Cin, Cout, weights=walnut_w1, use_bias=False),
            walnut.nn.layers.Linear(Cout, Cout, weights=walnut_w2, use_bias=False),
        ],
        -1,
    )
    walnut_module.training = True
    walnut_module.to_device("cuda")
    walnut_y = walnut_module(walnut_x)

    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    lin2 = torch.nn.Linear(Cout, Cout, bias=False)
    lin2.weight = torch_w2
    torch_parallel_modules = [lin1, lin2]
    torch_y = torch.cat([m(torch_x) for m in torch_parallel_modules], -1)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False, device="cuda")
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))

    assert all(results)
