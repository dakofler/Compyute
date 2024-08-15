"""Normalization module tests"""

import pytest
import torch
import torchtune

from compyute.nn import BatchNorm1D, BatchNorm2D, LayerNorm, RMSNorm
from tests.test_utils import get_random_floats, is_equal

SHAPE3D = (10, 20, 30)
SHAPE4D = (10, 20, 30, 40)


def test_batchnorm1d() -> None:
    """Test for the batchnorm 1d layer."""
    shape_x = SHAPE3D

    # init compyute module
    compyute_module = BatchNorm1D(shape_x[1])

    # init torch module
    torch_module = torch.nn.BatchNorm1d(shape_x[1])

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    with compyute_module.train():
        compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)
    assert is_equal(compyute_module.rmean, torch_module.running_mean)
    assert is_equal(compyute_module.rvar, torch_module.running_var)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    with compyute_module.train():
        compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)


def test_batchnorm2d() -> None:
    """Test for the batchnorm 2d layer."""
    shape_x = SHAPE4D

    # init compyute module
    compyute_module = BatchNorm2D(shape_x[1])

    # init torch module
    torch_module = torch.nn.BatchNorm2d(shape_x[1])

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    with compyute_module.train():
        compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)
    assert is_equal(compyute_module.rmean, torch_module.running_mean)
    assert is_equal(compyute_module.rvar, torch_module.running_var)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    with compyute_module.train():
        compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)


@pytest.mark.parametrize(
    "normalized_shape",
    [SHAPE3D[1:], SHAPE3D[2:]],
)
def test_layernorm(normalized_shape) -> None:
    """Test for the layernorm layer."""
    # init compyute module
    compyute_module = LayerNorm(normalized_shape)

    # init torch module
    torch_module = torch.nn.LayerNorm(normalized_shape)

    # forward
    compyute_x, torch_x = get_random_floats(SHAPE3D)
    with compyute_module.train():
        compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(SHAPE3D, torch_grad=False)
    with compyute_module.train():
        compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)


def test_rmsnorm() -> None:
    """Test for the rmsnorm layer."""
    normalized_shape = SHAPE3D[-1]

    # init compyute module
    compyute_module = RMSNorm((normalized_shape,), eps=1e-6)

    # init torch module
    torch_module = torchtune.modules.RMSNorm(normalized_shape)

    # forward
    compyute_x, torch_x = get_random_floats(SHAPE3D)
    with compyute_module.train():
        compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(SHAPE3D, torch_grad=False)
    with compyute_module.train():
        compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.scale.grad)
