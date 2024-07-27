"""Convolution module tests"""

import torch

from compyute.nn import AvgPooling2d, Convolution1d, Convolution2d, MaxPooling2d
from tests.test_utils import get_random_floats, get_random_params, is_equal

B, Cin, Cout, Y, X, K = (10, 3, 16, 15, 15, 5)


def test_conv1d_valid() -> None:
    """Test for the conv 1d layer using valid padding."""
    shape_x = (B, Cin, X)
    shape_w = (Cout, Cin, K)
    shape_b = (Cout,)
    strides = 1
    dilation = 1
    pad = "valid"

    # init parameters
    compyute_w, torch_w = get_random_params(shape_w)
    compyute_b, torch_b = get_random_params(shape_b)

    # init compyute module
    compyute_module = Convolution1d(Cin, Cout, K, pad, strides, dilation, training=True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b

    # init torch module
    torch_module = torch.nn.Conv1d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)


def test_conv1d_same() -> None:
    """Test for the conv 1d layer using same padding."""
    shape_x = (B, Cin, X)
    shape_w = (Cout, Cin, K)
    shape_b = (Cout,)
    strides = 1
    dilation = 1
    pad = "same"

    # init parameters
    compyute_w, torch_w = get_random_params(shape_w)
    compyute_b, torch_b = get_random_params(shape_b)

    # init compyute module
    compyute_module = Convolution1d(Cin, Cout, K, pad, strides, dilation, training=True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b

    # init torch module
    torch_module = torch.nn.Conv1d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)


def test_conv1d_valid_dilation2() -> None:
    """Test for the conv 1d layer using valid padding and dilations of 2."""
    shape_x = (B, Cin, X)
    shape_w = (Cout, Cin, K)
    shape_b = (Cout,)
    strides = 1
    dilation = 2
    pad = "valid"

    # init parameters
    compyute_w, torch_w = get_random_params(shape_w)
    compyute_b, torch_b = get_random_params(shape_b)

    # init compyute module
    compyute_module = Convolution1d(Cin, Cout, K, pad, strides, dilation, training=True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b

    # init torch module
    torch_module = torch.nn.Conv1d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)


def test_conv1d_valid_stride2() -> None:
    """Test for the conv 1d layer using valid padding and strides of 2."""
    shape_x = (B, Cin, X)
    shape_w = (Cout, Cin, K)
    shape_b = (Cout,)
    strides = 2
    dilation = 1
    pad = "valid"

    # init parameters
    compyute_w, torch_w = get_random_params(shape_w)
    compyute_b, torch_b = get_random_params(shape_b)

    # init compyute module
    compyute_module = Convolution1d(Cin, Cout, K, pad, strides, dilation, training=True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b

    # init torch module
    torch_module = torch.nn.Conv1d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)


def test_conv1d_same_dilation2() -> None:
    """Test for the conv 1d layer using same padding and dilations of 2."""
    shape_x = (B, Cin, X)
    shape_w = (Cout, Cin, K)
    shape_b = (Cout,)
    strides = 1
    dilation = 2
    pad = "same"

    # init parameters
    compyute_w, torch_w = get_random_params(shape_w)
    compyute_b, torch_b = get_random_params(shape_b)

    # init compyute module
    compyute_module = Convolution1d(Cin, Cout, K, pad, strides, dilation, training=True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b

    # init torch module
    torch_module = torch.nn.Conv1d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)


def test_conv2d_valid() -> None:
    """Test for the conv 2d layer using valid padding."""
    shape_x = (B, Cin, Y, X)
    shape_w = (Cout, Cin, K, K)
    shape_b = (Cout,)
    strides = 1
    dilation = 1
    pad = "valid"

    # init parameters
    compyute_w, torch_w = get_random_params(shape_w)
    compyute_b, torch_b = get_random_params(shape_b)

    compyute_module = Convolution2d(Cin, Cout, K, pad, strides, dilation, training=True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b

    torch_module = torch.nn.Conv2d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)


def test_conv2d_same() -> None:
    """Test for the conv 2d layer using same padding."""
    shape_x = (B, Cin, Y, X)
    shape_w = (Cout, Cin, K, K)
    shape_b = (Cout,)
    strides = 1
    dilation = 1
    pad = "same"

    # init parameters
    compyute_w, torch_w = get_random_params(shape_w)
    compyute_b, torch_b = get_random_params(shape_b)

    # init compyute module
    compyute_module = Convolution2d(Cin, Cout, K, pad, strides, dilation, training=True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b

    # init torch module
    torch_module = torch.nn.Conv2d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)  # error
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)


def test_conv2d_valid_stride2() -> None:
    """Test for the conv 2d layer using valid padding and strides of 2."""
    shape_x = (B, Cin, Y, X)
    shape_w = (Cout, Cin, K, K)
    shape_b = (Cout,)
    strides = 2
    dilation = 1
    pad = "valid"

    # init parameters
    compyute_w, torch_w = get_random_params(shape_w)
    compyute_b, torch_b = get_random_params(shape_b)

    # init compyute module
    compyute_module = Convolution2d(Cin, Cout, K, pad, strides, dilation, training=True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b

    # init torch module
    torch_module = torch.nn.Conv2d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)


def test_conv2d_valid_dilation2() -> None:
    """Test for the conv 2d layer using valid padding and dilations of 2."""
    shape_x = (B, Cin, Y, X)
    shape_w = (Cout, Cin, K, K)
    shape_b = (Cout,)
    strides = 1
    dilation = 2
    pad = "valid"

    # init parameters
    compyute_w, torch_w = get_random_params(shape_w)
    compyute_b, torch_b = get_random_params(shape_b)

    # init compyute module
    compyute_module = Convolution2d(Cin, Cout, K, pad, strides, dilation, training=True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b

    # init torch module
    torch_module = torch.nn.Conv2d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)


def test_conv2d_same_dilation2() -> None:
    """Test for the conv 2d layer using same padding and dilations of 2."""
    shape_x = (B, Cin, Y, X)
    shape_w = (Cout, Cin, K, K)
    shape_b = (Cout,)
    strides = 1
    dilation = 2
    pad = "same"

    # init parameters
    compyute_w, torch_w = get_random_params(shape_w)
    compyute_b, torch_b = get_random_params(shape_b)

    # init compyute module
    compyute_module = Convolution2d(Cin, Cout, K, pad, strides, dilation, training=True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b

    # init torch module
    torch_module = torch.nn.Conv2d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
    assert is_equal(compyute_module.b.grad, torch_module.bias.grad)


def test_maxpool2d() -> None:
    """Test for the maxpool layer."""
    shape_x = (B, Cout, Y, X)

    # init compyute module
    compyute_module = MaxPooling2d(training=True)

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch.nn.functional.max_pool2d(torch_x, 2)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)


def test_avgpool2d() -> None:
    """Test for the avgpool layer."""
    shape_x = (B, Cout, Y, X)

    # init compyute module
    compyute_module = AvgPooling2d(training=True)

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch.nn.functional.avg_pool2d(torch_x, (2, 2))
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
