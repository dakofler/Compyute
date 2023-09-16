"""Parameter layer tests"""

import torch
import walnut
from tests.test_utils import get_vals, get_params, validate


B, Cin, Cout, Y, X, K = (10, 10, 10, 10, 10, 3)


# Linear
def test_linear_cpu() -> None:
    results = []
    shape_x = (B, Cin)
    shape_w = (Cin, Cout)
    shape_b = (Cout,)

    # forward
    walnut_x, torch_x = get_vals(shape_x)
    walnut_w, torch_w = get_params(shape_w, T=True)
    walnut_b, torch_b = get_params(shape_b)

    walnut_module = walnut.nn.layers.Linear(Cin, Cout)
    walnut_module.training = True
    walnut_module.w = walnut_w
    walnut_module.b = walnut_b
    walnut_y = walnut_module(walnut_x)

    torch_module = torch.nn.Linear(Cin, Cout)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False)
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(walnut_dx, torch_x.grad))
    results.append(validate(walnut_module.w.grad, torch_module.weight.grad.T))
    results.append(validate(walnut_module.b.grad, torch_module.bias.grad))

    assert all(results)


def test_linear_cuda() -> None:
    results = []
    shape_x = (B, Cin)
    shape_w = (Cin, Cout)
    shape_b = (Cout,)

    # forward
    walnut_x, torch_x = get_vals(shape_x, device="cuda")
    walnut_w, torch_w = get_params(shape_w, T=True, device="cuda")
    walnut_b, torch_b = get_params(shape_b, device="cuda")

    walnut_module = walnut.nn.layers.Linear(Cin, Cout)
    walnut_module.training = True
    walnut_module.to_device("cuda")
    walnut_module.w = walnut_w
    walnut_module.b = walnut_b
    walnut_y = walnut_module(walnut_x)

    torch_module = torch.nn.Linear(Cin, Cout)
    torch_module.weight = torch_w
    torch_module.bias = torch_b
    torch_y = torch_module(torch_x)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False, device="cuda")
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(walnut_dx, torch_x.grad))
    results.append(validate(walnut_module.w.grad, torch_module.weight.grad.T))
    results.append(validate(walnut_module.b.grad, torch_module.bias.grad))

    assert all(results)


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


# Embedding
def test_embedding_cpu() -> None:
    results = []
    shape_w = (Cin, Cout)

    # forward
    walnut_x = walnut.randint((B, X), 0, Cin)
    torch_x = torch.from_numpy(walnut_x.data)
    walnut_w, torch_w = get_params(shape_w)

    walnut_module = walnut.nn.layers.Embedding(Cin, Cout)
    walnut_module.training = True
    walnut_module.w = walnut_w
    walnut_y = walnut_module(walnut_x)

    torch_module = torch.nn.Embedding(Cin, Cout)
    torch_module.weight = torch_w
    torch_y = torch_module(torch_x)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False)
    _ = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(walnut_module.w.grad, torch_module.weight.grad))

    assert all(results)


def test_embedding_cuda() -> None:
    results = []
    shape_w = (Cin, Cout)

    # forward
    walnut_x = walnut.randint((B, X), 0, Cin)
    torch_x = torch.from_numpy(walnut_x.data)
    walnut_x.to_device("cuda")
    walnut_w, torch_w = get_params(shape_w, device="cuda")

    walnut_module = walnut.nn.layers.Embedding(Cin, Cout)
    walnut_module.training = True
    walnut_module.to_device("cuda")
    walnut_module.w = walnut_w
    walnut_y = walnut_module(walnut_x)

    torch_module = torch.nn.Embedding(Cin, Cout)
    torch_module.weight = torch_w
    torch_y = torch_module(torch_x)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False, device="cuda")
    _ = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(walnut_module.w.grad, torch_module.weight.grad))

    assert all(results)
