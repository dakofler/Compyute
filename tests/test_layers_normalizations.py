"""Normalization layer tests"""

import torch

from src.compyute.nn import Batchnorm1d, Batchnorm2d, Layernorm
from tests.test_utils import get_random_floats, is_equal

SHAPE3D = (10, 20, 30)
SHAPE4D = (10, 20, 30, 40)


def test_batchnorm1d() -> None:
    """Test for the batchnorm 1d layer."""
    shape_x = SHAPE3D

    # init compyute module
    compyute_module = Batchnorm1d(shape_x[1], training=True)

    # init torch module
    torch_module = torch.nn.BatchNorm1d(shape_x[1])

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)
    assert is_equal(compyute_module.rmean, torch_module.running_mean)
    assert is_equal(compyute_module.rvar, torch_module.running_var)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)


def test_batchnorm2d() -> None:
    """Test for the batchnorm 2d layer."""
    shape_x = SHAPE4D

    # init compyute module
    compyute_module = Batchnorm2d(shape_x[1], training=True)

    # init torch module
    torch_module = torch.nn.BatchNorm2d(shape_x[1])

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)
    assert is_equal(compyute_module.rmean, torch_module.running_mean)
    assert is_equal(compyute_module.rvar, torch_module.running_var)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)


def test_layernorm() -> None:
    """Test for the layernorm layer."""
    shape_x = SHAPE3D

    # init compyute module
    compyute_module = Layernorm(shape_x[1:], training=True)

    # init torch module
    torch_module = torch.nn.LayerNorm(shape_x[1:])

    # forward
    compyute_x, torch_x = get_random_floats(SHAPE3D)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(SHAPE3D, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)
