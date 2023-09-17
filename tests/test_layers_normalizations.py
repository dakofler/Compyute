"""Normalization layer tests"""

import torch
import walnut
from tests.test_utils import get_vals, validate


SHAPE3D = (10, 10, 10)
SHAPE4D = (10, 10, 10, 10)


# Batchnorm
def test_batchnorm1d_cpu() -> None:
    results = []

    # forward
    walnut_x, torch_x = get_vals(SHAPE3D)
    walnut_module = walnut.nn.layers.Batchnorm(SHAPE3D[1])
    walnut_module.training = True
    walnut_y = walnut_module(walnut_x)
    torch_module = torch.nn.BatchNorm1d(SHAPE3D[1])
    torch_y = torch_module(torch_x)
    results.append(validate(walnut_y, torch_y))
    results.append(validate(walnut_module.rmean, torch_module.running_mean))
    results.append(validate(walnut_module.rvar, torch_module.running_var))

    # backward
    walnut_dy, torch_dy = get_vals(SHAPE3D, torch_grad=False)
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))
    results.append(validate(walnut_module.w.grad, torch_module.weight.grad))
    results.append(validate(walnut_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_batchnorm1d_cuda() -> None:
    if not walnut.cuda.is_available():
        pass
    results = []

    # forward
    walnut_x, torch_x = get_vals(SHAPE3D, device="cuda")
    walnut_module = walnut.nn.layers.Batchnorm(SHAPE3D[1])
    walnut_module.training = True
    walnut_module.to_device("cuda")
    walnut_y = walnut_module(walnut_x)
    torch_module = torch.nn.BatchNorm1d(SHAPE3D[1])
    torch_y = torch_module(torch_x)
    results.append(validate(walnut_y, torch_y))
    results.append(validate(walnut_module.rmean, torch_module.running_mean))
    results.append(validate(walnut_module.rvar, torch_module.running_var))

    # backward
    walnut_dy, torch_dy = get_vals(SHAPE3D, torch_grad=False, device="cuda")
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))
    results.append(validate(walnut_module.w.grad, torch_module.weight.grad))
    results.append(validate(walnut_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_batchnorm2d_cpu() -> None:
    results = []

    # forward
    walnut_x, torch_x = get_vals(SHAPE4D)
    walnut_module = walnut.nn.layers.Batchnorm(SHAPE4D[1])
    walnut_module.training = True
    walnut_y = walnut_module(walnut_x)
    torch_module = torch.nn.BatchNorm2d(SHAPE4D[1])
    torch_y = torch_module(torch_x)
    results.append(validate(walnut_y, torch_y))
    results.append(validate(walnut_module.rmean, torch_module.running_mean))
    results.append(validate(walnut_module.rvar, torch_module.running_var))

    # backward
    walnut_dy, torch_dy = get_vals(SHAPE4D, torch_grad=False)
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))
    results.append(validate(walnut_module.w.grad, torch_module.weight.grad))
    results.append(validate(walnut_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_batchnorm2d_cuda() -> None:
    if not walnut.cuda.is_available():
        pass
    results = []

    # forward
    walnut_x, torch_x = get_vals(SHAPE4D, device="cuda")
    walnut_module = walnut.nn.layers.Batchnorm(SHAPE4D[1])
    walnut_module.training = True
    walnut_module.to_device("cuda")
    walnut_y = walnut_module(walnut_x)
    torch_module = torch.nn.BatchNorm2d(SHAPE4D[1])
    torch_y = torch_module(torch_x)
    results.append(validate(walnut_y, torch_y))
    results.append(validate(walnut_module.rmean, torch_module.running_mean))
    results.append(validate(walnut_module.rvar, torch_module.running_var))

    # backward
    walnut_dy, torch_dy = get_vals(SHAPE4D, torch_grad=False, device="cuda")
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))
    results.append(validate(walnut_module.w.grad, torch_module.weight.grad))
    results.append(validate(walnut_module.b.grad, torch_module.bias.grad))

    assert all(results)


# Layernorm
def test_layernorm_cpu() -> None:
    results = []

    # forward
    walnut_x, torch_x = get_vals(SHAPE3D)
    walnut_module = walnut.nn.layers.Layernorm(SHAPE3D[1:])
    walnut_module.training = True
    walnut_y = walnut_module(walnut_x)
    torch_module = torch.nn.LayerNorm(SHAPE3D[1:])
    torch_y = torch_module(torch_x)
    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(SHAPE3D, torch_grad=False)
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))
    results.append(validate(walnut_module.w.grad, torch_module.weight.grad))
    results.append(validate(walnut_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_layernorm_cuda() -> None:
    if not walnut.cuda.is_available():
        pass
    results = []

    # forward
    walnut_x, torch_x = get_vals(SHAPE3D, device="cuda")
    walnut_module = walnut.nn.layers.Layernorm(SHAPE3D[1:])
    walnut_module.training = True
    walnut_module.to_device("cuda")
    walnut_y = walnut_module(walnut_x)
    torch_module = torch.nn.LayerNorm(SHAPE3D[1:])
    torch_y = torch_module(torch_x)
    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(SHAPE3D, torch_grad=False, device="cuda")
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))
    results.append(validate(walnut_module.w.grad, torch_module.weight.grad))
    results.append(validate(walnut_module.b.grad, torch_module.bias.grad))

    assert all(results)
