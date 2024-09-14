"""Convolution module tests"""

import pytest
import torch

from compyute.nn import AvgPooling2D, Convolution1D, Convolution2D, MaxPooling2D
from tests.utils import get_random_floats, get_random_params, is_close

conv1d_testdata = [
    ((16, 64, 32, 16, 5), "valid", 1, 1),
    ((16, 64, 32, 16, 5), "valid", 1, 2),
    ((16, 64, 32, 16, 5), "valid", 2, 1),
    ((16, 64, 32, 16, 5), "valid", 2, 2),
    ((16, 64, 32, 16, 5), "same", 1, 1),
    ((16, 64, 32, 16, 5), "same", 1, 2),
    ((32, 128, 64, 16, 3), "valid", 1, 1),
    ((32, 128, 64, 16, 3), "valid", 1, 2),
    ((32, 128, 64, 16, 3), "valid", 2, 1),
    ((32, 128, 64, 16, 3), "valid", 2, 2),
    ((32, 128, 64, 16, 3), "same", 1, 1),
    ((32, 128, 64, 16, 3), "same", 1, 2),
]

conv2d_testdata = [
    ((16, 3, 32, 28, 28, 5), "valid", 1, 1),
    ((16, 3, 32, 28, 28, 5), "valid", 1, 2),
    ((16, 3, 32, 28, 28, 5), "valid", 2, 1),
    ((16, 3, 32, 28, 28, 5), "valid", 2, 2),
    ((16, 3, 32, 28, 28, 5), "same", 1, 1),
    ((16, 3, 32, 28, 28, 5), "same", 1, 2),
    ((32, 1, 64, 32, 32, 3), "valid", 1, 1),
    ((32, 1, 64, 32, 32, 3), "valid", 1, 2),
    ((32, 1, 64, 32, 32, 3), "valid", 2, 1),
    ((32, 1, 64, 32, 32, 3), "valid", 2, 2),
    ((32, 1, 64, 32, 32, 3), "same", 1, 1),
    ((32, 1, 64, 32, 32, 3), "same", 1, 2),
]

pool_testdata = [
    ((16, 32, 28, 28), 2),
    ((16, 32, 28, 28), 3),
    ((32, 64, 32, 32), 2),
    ((32, 64, 32, 32), 3),
]


@pytest.mark.parametrize("shape,padding,strides,dilation", conv1d_testdata)
def test_conv1d(shape, padding, strides, dilation) -> None:
    """Test for the conv 1d layer."""

    B, Cin, Cout, X, K = shape
    shape_x = (B, Cin, X)
    shape_w = (Cout, Cin, K)
    shape_b = (Cout,)

    # init parameters
    compyute_w, torch_w = get_random_params(shape_w)
    compyute_b, torch_b = get_random_params(shape_b)

    # init compyute module
    compyute_module = Convolution1D(Cin, Cout, K, padding, strides, dilation)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b

    # init torch module
    torch_module = torch.nn.Conv1d(Cin, Cout, K, strides, padding, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)
    assert is_close(compyute_module.w.grad, torch_module.weight.grad)
    assert is_close(compyute_module.b.grad, torch_module.bias.grad)


@pytest.mark.parametrize("shape,padding,strides,dilation", conv2d_testdata)
def test_conv2d(shape, padding, strides, dilation) -> None:
    """Test for the conv 2d layer."""

    B, Cin, Cout, Y, X, K = shape
    shape_x = (B, Cin, Y, X)
    shape_w = (Cout, Cin, K, K)
    shape_b = (Cout,)

    # init parameters
    compyute_w, torch_w = get_random_params(shape_w)
    compyute_b, torch_b = get_random_params(shape_b)

    compyute_module = Convolution2D(Cin, Cout, K, padding, strides, dilation)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b

    torch_module = torch.nn.Conv2d(Cin, Cout, K, strides, padding, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)
    assert is_close(compyute_module.w.grad, torch_module.weight.grad)
    assert is_close(compyute_module.b.grad, torch_module.bias.grad, tol=1e-4)


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
