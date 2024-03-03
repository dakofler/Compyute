"""Convolutional layer tests"""

import torch
import compyute
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
    compyute_x, torch_x = get_vals(shape_x)
    compyute_w, torch_w = get_params(shape_w)
    compyute_b, torch_b = get_params(shape_b)

    compyute_module = compyute.nn.layers.Convolution1d(
        Cin, Cout, K, pad, strides, dilation
    )
    compyute_module.training = True
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Conv1d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_conv1d_cuda() -> None:
    if not compyute.engine.gpu_available():
        pass
    results = []
    shape_x = (B, Cin, X)
    shape_w = (Cout, Cin, K)
    shape_b = (Cout,)

    strides = 1
    dilation = 1
    pad = "valid"

    # forward
    compyute_x, torch_x = get_vals(shape_x, device="cuda")
    compyute_w, torch_w = get_params(shape_w, device="cuda")
    compyute_b, torch_b = get_params(shape_b, device="cuda")

    compyute_module = compyute.nn.layers.Convolution1d(
        Cin, Cout, K, pad, strides, dilation
    )
    compyute_module.training = True
    compyute_module.to_device("cuda")
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Conv1d(Cin, Cout, K, strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False, device="cuda")
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_dx, torch_x.grad))
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

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
    compyute_x, torch_x = get_vals(shape_x)
    compyute_w, torch_w = get_params(shape_w)
    compyute_b, torch_b = get_params(shape_b)

    compyute_module = compyute.nn.layers.Convolution2d(
        Cin, Cout, (K, K), pad, strides, dilation
    )
    compyute_module.training = True
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Conv2d(Cin, Cout, (K, K), strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_dx, torch_x.grad))
    # inaccuracy?
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad, 1e-3))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_conv2d_cuda() -> None:
    if not compyute.engine.gpu_available():
        pass
    results = []
    shape_x = (B, Cin, Y, X)
    shape_w = (Cout, Cin, K, K)
    shape_b = (Cout,)

    strides = 1
    dilation = 1
    pad = "valid"

    # forward
    compyute_x, torch_x = get_vals(shape_x, device="cuda")
    compyute_w, torch_w = get_params(shape_w, device="cuda")
    compyute_b, torch_b = get_params(shape_b, device="cuda")

    compyute_module = compyute.nn.layers.Convolution2d(
        Cin, Cout, (K, K), pad, strides, dilation
    )
    compyute_module.training = True
    compyute_module.to_device("cuda")
    compyute_module.w = compyute_w
    compyute_module.b = compyute_b
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Conv2d(Cin, Cout, (K, K), strides, pad, dilation)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False, device="cuda")
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_dx, torch_x.grad))
    # inaccuracy?
    results.append(validate(compyute_module.w.grad, torch_module.weight.grad, 1e-3))
    results.append(validate(compyute_module.b.grad, torch_module.bias.grad))

    assert all(results)


# Maxpool2d
def test_maxpool2d_cpu() -> None:
    results = []
    shape_x = (B, Cout, Y, X)

    # forward
    compyute_x, torch_x = get_vals(shape_x)
    compyute_module = compyute.nn.layers.MaxPooling2d()
    compyute_module.training = True
    compyute_y = compyute_module(compyute_x)
    torch_y = torch.nn.functional.max_pool2d(torch_x, (2, 2))
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_maxpool2d_cuda() -> None:
    if not compyute.engine.gpu_available():
        pass
    results = []
    shape_x = (B, Cout, Y, X)

    # forward
    compyute_x, torch_x = get_vals(shape_x, device="cuda")
    compyute_module = compyute.nn.layers.MaxPooling2d()
    compyute_module.training = True
    compyute_module.to_device("cuda")
    compyute_y = compyute_module(compyute_x)
    torch_y = torch.nn.functional.max_pool2d(torch_x, (2, 2))
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False, device="cuda")
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)
