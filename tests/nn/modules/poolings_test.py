"""Pooling module tests"""

import pytest
import torch

from compyute.nn import AvgPooling2D, MaxPooling2D, Upsample2D
from tests.utils import get_random_floats, is_close

pool_testdata = [
    ((16, 32, 28, 28), 2),
    ((16, 32, 28, 28), 3),
    ((32, 64, 32, 32), 2),
    ((32, 64, 32, 32), 3),
]


@pytest.mark.parametrize("shape,kernel_size", pool_testdata)
def test_upsample2d(shape, kernel_size) -> None:
    """Test for the upsample layer."""

    # init compyute module
    compyute_module = Upsample2D(kernel_size)

    # forward
    compyute_x, torch_x = get_random_floats(shape)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch.nn.functional.interpolate(torch_x, scale_factor=kernel_size)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)


@pytest.mark.parametrize("shape,kernel_size", pool_testdata)
def test_maxpool2d(shape, kernel_size) -> None:
    """Test for the maxpool layer."""

    # init compyute module
    compyute_module = MaxPooling2D(kernel_size)

    # forward
    compyute_x, torch_x = get_random_floats(shape)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch.nn.functional.max_pool2d(torch_x, kernel_size)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)


@pytest.mark.parametrize("shape,kernel_size", pool_testdata)
def test_avgpool2d(shape, kernel_size) -> None:
    """Test for the avgpool layer."""

    # init compyute module
    compyute_module = AvgPooling2D(kernel_size)

    # forward
    compyute_x, torch_x = get_random_floats(shape)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch.nn.functional.avg_pool2d(torch_x, kernel_size)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)
