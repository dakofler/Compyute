"""Activation layer tests"""

import torch.nn.functional as F
import walnut
from tests.test_utils import get_vals, validate


SHAPE = (10, 10, 10)


# ReLU
def test_relu_cpu() -> None:
    results = []

    # forward
    walnut_x, torch_x = get_vals(SHAPE)
    module = walnut.nn.layers.ReLU()
    module.training = True
    walnut_y = module(walnut_x)
    torch_y = F.relu(torch_x)
    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(SHAPE, torch_grad=False)
    walnut_dx = module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))

    assert all(results)


def test_relu_cuda() -> None:
    if not walnut.cuda.is_available():
        pass
    results = []

    # forward
    walnut_x, torch_x = get_vals(SHAPE, device="cuda")
    module = walnut.nn.layers.ReLU()
    module.training = True
    walnut_y = module(walnut_x)
    torch_y = F.relu(torch_x)
    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(SHAPE, torch_grad=False, device="cuda")
    walnut_dx = module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))

    assert all(results)


# Tanh
def test_tanh_cpu() -> None:
    results = []

    # forward
    walnut_x, torch_x = get_vals(SHAPE)
    module = walnut.nn.layers.Tanh()
    module.training = True
    walnut_y = module(walnut_x)
    torch_y = F.tanh(torch_x)
    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(SHAPE, torch_grad=False)
    walnut_dx = module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))

    assert all(results)


def test_tanh_cuda() -> None:
    if not walnut.cuda.is_available():
        pass
    results = []

    # forward
    walnut_x, torch_x = get_vals(SHAPE, device="cuda")
    module = walnut.nn.layers.Tanh()
    module.training = True
    walnut_y = module(walnut_x)
    torch_y = F.tanh(torch_x)
    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(SHAPE, torch_grad=False, device="cuda")
    walnut_dx = module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))

    assert all(results)


# Sigmoid
def test_sigmoid_cpu() -> None:
    results = []

    # forward
    walnut_x, torch_x = get_vals(SHAPE)
    module = walnut.nn.layers.Sigmoid()
    module.training = True
    walnut_y = module(walnut_x)
    torch_y = F.sigmoid(torch_x)
    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(SHAPE, torch_grad=False)
    walnut_dx = module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))

    assert all(results)


def test_sigmoid_cuda() -> None:
    if not walnut.cuda.is_available():
        pass
    results = []

    # forward
    walnut_x, torch_x = get_vals(SHAPE, device="cuda")
    module = walnut.nn.layers.Sigmoid()
    module.training = True
    walnut_y = module(walnut_x)
    torch_y = F.sigmoid(torch_x)
    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(SHAPE, torch_grad=False, device="cuda")
    walnut_dx = module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(walnut_dx, torch_x.grad))

    assert all(results)
