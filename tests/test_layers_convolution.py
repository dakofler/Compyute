"""Convolutional layer tests"""

import torch
import walnut
from tests.test_utils import get_vals, get_params, validate


B, Cin, Cout, Y, X, K = (10, 10, 10, 10, 10, 3)


# Conv1d
def test_conv1d_cpu() -> None:
    results = []
    shape_x = (B, Cin, X)
    shape_w = (Cout, Cin, K)
    shape_b = (Cout,)

    strides = 1
    dilation = 1
    pad = "valid"

    # forward
    walnut_x, torch_x = get_vals(shape_x)
    walnut_w, torch_w = get_params(shape_w)
    walnut_b, torch_b = get_params(shape_b)

    walnut_module = walnut.nn.layers.Convolution1d(Cin, Cout, K, pad, strides, dilation)
    walnut_module.training = True
    walnut_module.w = walnut_w
    walnut_module.b = walnut_b
    walnut_y = walnut_module(walnut_x)

    torch_module = torch.nn.Conv1d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False)
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(walnut_dx, torch_x.grad))
    results.append(validate(walnut_module.w.grad, torch_module.weight.grad))
    results.append(validate(walnut_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_conv1d_cuda() -> None:
    results = []
    shape_x = (B, Cin, X)
    shape_w = (Cout, Cin, K)
    shape_b = (Cout,)

    strides = 1
    dilation = 1
    pad = "valid"

    # forward
    walnut_x, torch_x = get_vals(shape_x, device="cuda")
    walnut_w, torch_w = get_params(shape_w, device="cuda")
    walnut_b, torch_b = get_params(shape_b, device="cuda")

    walnut_module = walnut.nn.layers.Convolution1d(Cin, Cout, K, pad, strides, dilation)
    walnut_module.training = True
    walnut_module.to_device("cuda")
    walnut_module.w = walnut_w
    walnut_module.b = walnut_b
    walnut_y = walnut_module(walnut_x)

    torch_module = torch.nn.Conv1d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False, device="cuda")
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(walnut_dx, torch_x.grad))
    results.append(validate(walnut_module.w.grad, torch_module.weight.grad))
    results.append(validate(walnut_module.b.grad, torch_module.bias.grad))

    assert all(results)


# Conv2d
def test_conv2d_cpu() -> None:
    results = []
    shape_x = (B, Cin, Y, X)
    shape_w = (Cout, Cin, K, K)
    shape_b = (Cout,)

    strides = 1
    dilation = 1
    pad = "valid"

    # forward
    walnut_x, torch_x = get_vals(shape_x)
    walnut_w, torch_w = get_params(shape_w)
    walnut_b, torch_b = get_params(shape_b)

    walnut_module = walnut.nn.layers.Convolution2d(
        Cin, Cout, (K, K), pad, strides, dilation
    )
    walnut_module.training = True
    walnut_module.w = walnut_w
    walnut_module.b = walnut_b
    walnut_y = walnut_module(walnut_x)

    torch_module = torch.nn.Conv2d(Cin, Cout, (K, K), strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False)
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(walnut_dx, torch_x.grad))
    # inaccuracy?
    results.append(validate(walnut_module.w.grad, torch_module.weight.grad, 1e-3))
    results.append(validate(walnut_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_conv2d_cuda() -> None:
    results = []
    shape_x = (B, Cin, Y, X)
    shape_w = (Cout, Cin, K, K)
    shape_b = (Cout,)

    strides = 1
    dilation = 1
    pad = "valid"

    # forward
    walnut_x, torch_x = get_vals(shape_x, device="cuda")
    walnut_w, torch_w = get_params(shape_w, device="cuda")
    walnut_b, torch_b = get_params(shape_b, device="cuda")

    walnut_module = walnut.nn.layers.Convolution2d(
        Cin, Cout, (K, K), pad, strides, dilation
    )
    walnut_module.training = True
    walnut_module.to_device("cuda")
    walnut_module.w = walnut_w
    walnut_module.b = walnut_b
    walnut_y = walnut_module(walnut_x)

    torch_module = torch.nn.Conv2d(Cin, Cout, (K, K), strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False, device="cuda")
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(walnut_dx, torch_x.grad))
    # inaccuracy?
    results.append(validate(walnut_module.w.grad, torch_module.weight.grad, 1e-3))
    results.append(validate(walnut_module.b.grad, torch_module.bias.grad))

    assert all(results)


# Maxpool2d
def test_maxpool2d_cpu() -> None:
    results = []
    shape_x = (B, Cout, Y, X)

    # forward
    walnut_x, torch_x = get_vals(shape_x)
    walnut_module = walnut.nn.layers.MaxPooling2d()
    walnut_module.training = True
    walnut_y = walnut_module(walnut_x)
    torch_y = torch.nn.functional.max_pool2d(torch_x, (2, 2))
    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False)
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))

    assert all(results)


def test_maxpool2d_cuda() -> None:
    results = []
    shape_x = (B, Cout, Y, X)

    # forward
    walnut_x, torch_x = get_vals(shape_x, device="cuda")
    walnut_module = walnut.nn.layers.MaxPooling2d()
    walnut_module.training = True
    walnut_module.to_device("cuda")
    walnut_y = walnut_module(walnut_x)
    torch_y = torch.nn.functional.max_pool2d(torch_x, (2, 2))
    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False, device="cuda")
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))

    assert all(results)
