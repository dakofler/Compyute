"""Normalization layer tests"""

import torch

from src.compyute.nn import Batchnorm1d, Batchnorm2d, Layernorm
from tests.test_utils import get_vals_float, validate

SHAPE3D = (10, 20, 30)
SHAPE4D = (10, 20, 30, 40)


def test_batchnorm1d() -> None:
    """Test for the batchnorm 1d layer."""
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE3D)
    compyute_module = Batchnorm1d(SHAPE3D[1])
    compyute_module.set_training(True)
    compyute_y = compyute_module(compyute_x)
    torch_module = torch.nn.BatchNorm1d(SHAPE3D[1])
    torch_y = torch_module(torch_x)
    results.append(validate(compyute_y, torch_y))
    results.append(validate(compyute_module.rmean, torch_module.running_mean))
    results.append(validate(compyute_module.rvar, torch_module.running_var))

    # backward
    compyute_dy, torch_dy = get_vals_float(SHAPE3D, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_batchnorm2d() -> None:
    """Test for the batchnorm 2d layer."""
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE4D)
    compyute_module = Batchnorm2d(SHAPE4D[1])
    compyute_module.set_training(True)
    compyute_y = compyute_module(compyute_x)
    torch_module = torch.nn.BatchNorm2d(SHAPE4D[1])
    torch_y = torch_module(torch_x)
    results.append(validate(compyute_y, torch_y))
    results.append(validate(compyute_module.rmean, torch_module.running_mean))
    results.append(validate(compyute_module.rvar, torch_module.running_var))

    # backward
    compyute_dy, torch_dy = get_vals_float(SHAPE4D, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_layernorm() -> None:
    """Test for the layernorm layer."""
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE3D)
    compyute_module = Layernorm(SHAPE3D[1:])
    compyute_module.set_training(True)
    compyute_y = compyute_module(compyute_x)
    torch_module = torch.nn.LayerNorm(SHAPE3D[1:])
    torch_y = torch_module(torch_x)
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(SHAPE3D, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)
