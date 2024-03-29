"""Normalization layer tests"""

import torch
import compyute
from tests.test_utils import get_vals_float, validate


SHAPE3D = (10, 20, 30)
SHAPE4D = (10, 20, 30, 40)


def test_batchnorm1d() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE3D)
    compyute_module = compyute.nn.layers.Batchnorm1d(SHAPE3D[1])
    compyute_module.training = True
    compyute_y = compyute_module(compyute_x)
    torch_module = torch.nn.BatchNorm1d(SHAPE3D[1])
    torch_y = torch_module(torch_x)
    results.append(validate(compyute_y, torch_y))
    results.append(validate(compyute_module.rmean, torch_module.running_mean))
    results.append(validate(compyute_module.rvar, torch_module.running_var))

    # backward
    compyute_dy, torch_dy = get_vals_float(SHAPE3D, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad, tol=1e-4))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_batchnorm2d() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE4D)
    compyute_module = compyute.nn.layers.Batchnorm2d(SHAPE4D[1])
    compyute_module.training = True
    compyute_y = compyute_module(compyute_x)
    torch_module = torch.nn.BatchNorm2d(SHAPE4D[1])
    torch_y = torch_module(torch_x)
    results.append(validate(compyute_y, torch_y))
    results.append(validate(compyute_module.rmean, torch_module.running_mean))
    results.append(validate(compyute_module.rvar, torch_module.running_var))

    # backward
    compyute_dy, torch_dy = get_vals_float(SHAPE4D, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad, tol=1e-3))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_layernorm() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE3D)
    compyute_module = compyute.nn.layers.Layernorm(SHAPE3D[1:])
    compyute_module.training = True
    compyute_y = compyute_module(compyute_x)
    torch_module = torch.nn.LayerNorm(SHAPE3D[1:])
    torch_y = torch_module(torch_x)
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(SHAPE3D, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)
