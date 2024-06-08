"""Convolutional layer tests"""

import torch

from src.compyute.nn import AvgPooling2d, Convolution1d, Convolution2d, MaxPooling2d
from tests.test_utils import get_params, get_vals_float, validate

B, Cin, Cout, Y, X, K = (10, 3, 16, 15, 15, 5)


def test_conv1d_valid() -> None:
    """Test for the conv 1d layer using valid padding."""
    results = []
    shape_x = (B, Cin, X)
    shape_w = (Cout, Cin, K)
    shape_b = (Cout,)

    strides = 1
    dilation = 1
    pad = "valid"

    # forward
    compyute_x, torch_x = get_vals_float(shape_x)
    compyute_w, torch_w = get_params(shape_w)
    compyute_b, torch_b = get_params(shape_b)

    compyute_module = Convolution1d(Cin, Cout, K, pad, strides, dilation)
    compyute_module.set_training(True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Conv1d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_conv1d_same() -> None:
    """Test for the conv 1d layer using same padding."""
    results = []
    shape_x = (B, Cin, X)
    shape_w = (Cout, Cin, K)
    shape_b = (Cout,)

    strides = 1
    dilation = 1
    pad = "same"

    # forward
    compyute_x, torch_x = get_vals_float(shape_x)
    compyute_w, torch_w = get_params(shape_w)
    compyute_b, torch_b = get_params(shape_b)

    compyute_module = Convolution1d(Cin, Cout, K, pad, strides, dilation)
    compyute_module.set_training(True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Conv1d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_conv1d_valid_dilation2() -> None:
    """Test for the conv 1d layer using valid padding and dilations of 2."""
    results = []
    shape_x = (B, Cin, X)
    shape_w = (Cout, Cin, K)
    shape_b = (Cout,)

    strides = 1
    dilation = 2
    pad = "valid"

    # forward
    compyute_x, torch_x = get_vals_float(shape_x)
    compyute_w, torch_w = get_params(shape_w)
    compyute_b, torch_b = get_params(shape_b)

    compyute_module = Convolution1d(Cin, Cout, K, pad, strides, dilation)
    compyute_module.set_training(True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Conv1d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))  # error
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_conv1d_valid_stride2() -> None:
    """Test for the conv 1d layer using valid padding and strides of 2."""
    results = []
    shape_x = (B, Cin, X)
    shape_w = (Cout, Cin, K)
    shape_b = (Cout,)

    strides = 2
    dilation = 1
    pad = "valid"

    # forward
    compyute_x, torch_x = get_vals_float(shape_x)
    compyute_w, torch_w = get_params(shape_w)
    compyute_b, torch_b = get_params(shape_b)

    compyute_module = Convolution1d(Cin, Cout, K, pad, strides, dilation)
    compyute_module.set_training(True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Conv1d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))  # error
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_conv1d_same_dilation2() -> None:
    """Test for the conv 1d layer using same padding and dilations of 2."""
    results = []
    shape_x = (B, Cin, X)
    shape_w = (Cout, Cin, K)
    shape_b = (Cout,)

    strides = 1
    dilation = 2
    pad = "same"

    # forward
    compyute_x, torch_x = get_vals_float(shape_x)
    compyute_w, torch_w = get_params(shape_w)
    compyute_b, torch_b = get_params(shape_b)

    compyute_module = Convolution1d(Cin, Cout, K, pad, strides, dilation)
    compyute_module.set_training(True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Conv1d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))  # error
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_conv2d_valid() -> None:
    """Test for the conv 2d layer using valid padding."""
    results = []
    shape_x = (B, Cin, Y, X)
    shape_w = (Cout, Cin, K, K)
    shape_b = (Cout,)

    strides = 1
    dilation = 1
    pad = "valid"

    # forward
    compyute_x, torch_x = get_vals_float(shape_x)
    compyute_w, torch_w = get_params(shape_w)
    compyute_b, torch_b = get_params(shape_b)

    compyute_module = Convolution2d(Cin, Cout, K, pad, strides, dilation)
    compyute_module.set_training(True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Conv2d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_conv2d_same() -> None:
    """Test for the conv 2d layer using same padding."""
    results = []
    shape_x = (B, Cin, Y, X)
    shape_w = (Cout, Cin, K, K)
    shape_b = (Cout,)

    strides = 1
    dilation = 1
    pad = "same"

    # forward
    compyute_x, torch_x = get_vals_float(shape_x)
    compyute_w, torch_w = get_params(shape_w)
    compyute_b, torch_b = get_params(shape_b)

    compyute_module = Convolution2d(Cin, Cout, K, pad, strides, dilation)
    compyute_module.set_training(True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Conv2d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))  # error
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_conv2d_valid_stride2() -> None:
    """Test for the conv 2d layer using valid padding and strides of 2."""
    results = []
    shape_x = (B, Cin, Y, X)
    shape_w = (Cout, Cin, K, K)
    shape_b = (Cout,)

    strides = 2
    dilation = 1
    pad = "valid"

    # forward
    compyute_x, torch_x = get_vals_float(shape_x)
    compyute_w, torch_w = get_params(shape_w)
    compyute_b, torch_b = get_params(shape_b)

    compyute_module = Convolution2d(Cin, Cout, K, pad, strides, dilation)
    compyute_module.set_training(True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Conv2d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_conv2d_valid_dilation2() -> None:
    """Test for the conv 2d layer using valid padding and dilations of 2."""
    results = []
    shape_x = (B, Cin, Y, X)
    shape_w = (Cout, Cin, K, K)
    shape_b = (Cout,)

    strides = 1
    dilation = 2
    pad = "valid"

    # forward
    compyute_x, torch_x = get_vals_float(shape_x)
    compyute_w, torch_w = get_params(shape_w)
    compyute_b, torch_b = get_params(shape_b)

    compyute_module = Convolution2d(Cin, Cout, K, pad, strides, dilation)
    compyute_module.set_training(True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Conv2d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_conv2d_same_dilation2() -> None:
    """Test for the conv 2d layer using same padding and dilations of 2."""
    results = []
    shape_x = (B, Cin, Y, X)
    shape_w = (Cout, Cin, K, K)
    shape_b = (Cout,)

    strides = 1
    dilation = 2
    pad = "same"

    # forward
    compyute_x, torch_x = get_vals_float(shape_x)
    compyute_w, torch_w = get_params(shape_w)
    compyute_b, torch_b = get_params(shape_b)

    compyute_module = Convolution2d(Cin, Cout, K, pad, strides, dilation)
    compyute_module.set_training(True)
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Conv2d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_maxpool2d() -> None:
    """Test for the maxpool layer."""
    results = []
    shape_x = (B, Cout, Y, X)

    # forward
    compyute_x, torch_x = get_vals_float(shape_x)
    compyute_module = MaxPooling2d()
    compyute_module.set_training(True)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch.nn.functional.max_pool2d(torch_x, 2)
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_avgpool2d() -> None:
    """Test for the avgpool layer."""
    results = []
    shape_x = (B, Cout, Y, X)

    # forward
    compyute_x, torch_x = get_vals_float(shape_x)
    compyute_module = AvgPooling2d()
    compyute_module.set_training(True)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch.nn.functional.avg_pool2d(torch_x, (2, 2))
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)
