"""Activation layer tests"""

import torch.nn.functional as F
import compyute
from tests.test_utils import get_vals, validate


SHAPE = (10, 20, 30)


# ReLU
def test_relu_cpu() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_vals(SHAPE)
    module = compyute.nn.layers.ReLU()
    module.training = True
    compyute_y = module(compyute_x)
    torch_y = F.relu(torch_x)
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(SHAPE, torch_grad=False)
    compyute_dx = module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_relu_cuda() -> None:
    if not compyute.engine.gpu_available():
        pass
    results = []

    # forward
    compyute_x, torch_x = get_vals(SHAPE, device="cuda")
    module = compyute.nn.layers.ReLU()
    module.training = True
    compyute_y = module(compyute_x)
    torch_y = F.relu(torch_x)
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(SHAPE, torch_grad=False, device="cuda")
    compyute_dx = module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


# Leaky ReLU
def test_leaky_relu_cpu() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_vals(SHAPE)
    module = compyute.nn.layers.LeakyReLU(alpha=0.01)
    module.training = True
    compyute_y = module(compyute_x)
    torch_y = F.leaky_relu(torch_x, negative_slope=0.01)
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(SHAPE, torch_grad=False)
    compyute_dx = module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_leaky_relu_cuda() -> None:
    if not compyute.engine.gpu_available():
        pass
    results = []

    # forward
    compyute_x, torch_x = get_vals(SHAPE, device="cuda")
    module = compyute.nn.layers.LeakyReLU(alpha=0.01)
    module.training = True
    compyute_y = module(compyute_x)
    torch_y = F.leaky_relu(torch_x, negative_slope=0.01)
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(SHAPE, torch_grad=False, device="cuda")
    compyute_dx = module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


# GELU
def test_gelu_cpu() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_vals(SHAPE)
    module = compyute.nn.layers.GELU()
    module.training = True
    compyute_y = module(compyute_x)
    torch_y = F.gelu(torch_x, approximate="tanh")
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(SHAPE, torch_grad=False)
    compyute_dx = module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_gelu_cuda() -> None:
    if not compyute.engine.gpu_available():
        pass
    results = []

    # forward
    compyute_x, torch_x = get_vals(SHAPE, device="cuda")
    module = compyute.nn.layers.GELU()
    module.training = True
    compyute_y = module(compyute_x)
    torch_y = F.gelu(torch_x, approximate="tanh")
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(SHAPE, torch_grad=False, device="cuda")
    compyute_dx = module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


# Tanh
def test_tanh_cpu() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_vals(SHAPE)
    module = compyute.nn.layers.Tanh()
    module.training = True
    compyute_y = module(compyute_x)
    torch_y = F.tanh(torch_x)
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(SHAPE, torch_grad=False)
    compyute_dx = module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_tanh_cuda() -> None:
    if not compyute.engine.gpu_available():
        pass
    results = []

    # forward
    compyute_x, torch_x = get_vals(SHAPE, device="cuda")
    module = compyute.nn.layers.Tanh()
    module.training = True
    compyute_y = module(compyute_x)
    torch_y = F.tanh(torch_x)
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(SHAPE, torch_grad=False, device="cuda")
    compyute_dx = module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


# Sigmoid
def test_sigmoid_cpu() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_vals(SHAPE)
    module = compyute.nn.layers.Sigmoid()
    module.training = True
    compyute_y = module(compyute_x)
    torch_y = F.sigmoid(torch_x)
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(SHAPE, torch_grad=False)
    compyute_dx = module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_sigmoid_cuda() -> None:
    if not compyute.engine.gpu_available():
        pass
    results = []

    # forward
    compyute_x, torch_x = get_vals(SHAPE, device="cuda")
    module = compyute.nn.layers.Sigmoid()
    module.training = True
    compyute_y = module(compyute_x)
    torch_y = F.sigmoid(torch_x)
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(SHAPE, torch_grad=False, device="cuda")
    compyute_dx = module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)
