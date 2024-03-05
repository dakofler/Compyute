"""Containers tests"""

import torch
import compyute
from tests.test_utils import get_vals, get_params, validate


B, Cin, Cout = (10, 20, 30)


# Sequential container
def test_sequential_container_cpu() -> None:
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cin, Cout)
    w2_shape = (Cout, Cout)

    # forward
    compyute_x, torch_x = get_vals(x_shape)
    compyute_w1, torch_w1 = get_params(w1_shape, T=True)
    compyute_w2, torch_w2 = get_params(w2_shape, T=True)

    compyute_module = compyute.nn.containers.Sequential(
        [
            compyute.nn.layers.Linear(Cin, Cout, weights=compyute_w1, use_bias=False),
            compyute.nn.layers.Linear(Cout, Cout, weights=compyute_w2, use_bias=False),
        ]
    )
    compyute_module.training = True
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
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


# Sequential container
def test_sequential_container_cuda() -> None:
    if not compyute.engine.gpu_available():
        pass
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cin, Cout)
    w2_shape = (Cout, Cout)

    # forward
    compyute_x, torch_x = get_vals(x_shape, device="cuda")
    compyute_w1, torch_w1 = get_params(w1_shape, T=True, device="cuda")
    compyute_w2, torch_w2 = get_params(w2_shape, T=True, device="cuda")

    compyute_module = compyute.nn.containers.Sequential(
        [
            compyute.nn.layers.Linear(Cin, Cout, weights=compyute_w1, use_bias=False),
            compyute.nn.layers.Linear(Cout, Cout, weights=compyute_w2, use_bias=False),
        ]
    )
    compyute_module.training = True
    compyute_module.to_device("cuda")
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
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False, device="cuda")
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


# Parallel concat container
def test_parallel_concat_container_cpu() -> None:
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cin, Cout)
    w2_shape = (Cin, Cout)

    # forward
    compyute_x, torch_x = get_vals(x_shape)
    compyute_w1, torch_w1 = get_params(w1_shape, T=True)
    compyute_w2, torch_w2 = get_params(w2_shape, T=True)

    compyute_module = compyute.nn.containers.ParallelConcat(
        [
            compyute.nn.layers.Linear(Cin, Cout, weights=compyute_w1, use_bias=False),
            compyute.nn.layers.Linear(Cin, Cout, weights=compyute_w2, use_bias=False),
        ],
        -1,
    )
    compyute_module.training = True
    compyute_y = compyute_module(compyute_x)

    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    lin2 = torch.nn.Linear(Cin, Cout, bias=False)
    lin2.weight = torch_w2
    torch_parallel_modules = [lin1, lin2]
    torch_y = torch.cat([m(torch_x) for m in torch_parallel_modules], -1)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_parallel_concat_container_cuda() -> None:
    if not compyute.engine.gpu_available():
        pass
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cin, Cout)
    w2_shape = (Cin, Cout)

    # forward
    compyute_x, torch_x = get_vals(x_shape, device="cuda")
    compyute_w1, torch_w1 = get_params(w1_shape, T=True, device="cuda")
    compyute_w2, torch_w2 = get_params(w2_shape, T=True, device="cuda")

    compyute_module = compyute.nn.containers.ParallelConcat(
        [
            compyute.nn.layers.Linear(Cin, Cout, weights=compyute_w1, use_bias=False),
            compyute.nn.layers.Linear(Cin, Cout, weights=compyute_w2, use_bias=False),
        ],
        -1,
    )
    compyute_module.training = True
    compyute_module.to_device("cuda")
    compyute_y = compyute_module(compyute_x)

    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    lin2 = torch.nn.Linear(Cin, Cout, bias=False)
    lin2.weight = torch_w2
    torch_parallel_modules = [lin1, lin2]
    torch_y = torch.cat([m(torch_x) for m in torch_parallel_modules], -1)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False, device="cuda")
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


# Parallel add container
def test_parallel_add_container_cpu() -> None:
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cin, Cout)
    w2_shape = (Cin, Cout)

    # forward
    compyute_x, torch_x = get_vals(x_shape)
    compyute_w1, torch_w1 = get_params(w1_shape, T=True)
    compyute_w2, torch_w2 = get_params(w2_shape, T=True)

    compyute_module = compyute.nn.containers.ParallelAdd(
        [
            compyute.nn.layers.Linear(Cin, Cout, weights=compyute_w1, use_bias=False),
            compyute.nn.layers.Linear(Cin, Cout, weights=compyute_w2, use_bias=False),
        ]
    )
    compyute_module.training = True
    compyute_y = compyute_module(compyute_x)

    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    lin2 = torch.nn.Linear(Cin, Cout, bias=False)
    lin2.weight = torch_w2
    torch_y = lin1(torch_x) + lin2(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_parallel_add_container_cuda() -> None:
    if not compyute.engine.gpu_available():
        pass
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cin, Cout)
    w2_shape = (Cin, Cout)

    # forward
    compyute_x, torch_x = get_vals(x_shape, device="cuda")
    compyute_w1, torch_w1 = get_params(w1_shape, T=True, device="cuda")
    compyute_w2, torch_w2 = get_params(w2_shape, T=True, device="cuda")

    compyute_module = compyute.nn.containers.ParallelAdd(
        [
            compyute.nn.layers.Linear(Cin, Cout, weights=compyute_w1, use_bias=False),
            compyute.nn.layers.Linear(Cin, Cout, weights=compyute_w2, use_bias=False),
        ]
    )
    compyute_module.training = True
    compyute_module.to_device("cuda")
    compyute_y = compyute_module(compyute_x)

    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    lin2 = torch.nn.Linear(Cin, Cout, bias=False)
    lin2.weight = torch_w2
    torch_y = lin1(torch_x) + lin2(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False, device="cuda")
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)
