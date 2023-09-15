"""Activation layer tests"""

import torch.nn.functional as F
import walnut
from tests.test_utils import get_vals, validate


SHAPE = (10, 10, 10)


# ReLU
def test_relu_y_cpu() -> None:
    walnut_x, torch_x = get_vals(SHAPE)
    module = walnut.nn.layers.ReLU()
    module.training = True
    walnut_y = module(walnut_x)
    torch_y = F.relu(torch_x)
    assert validate(walnut_y, torch_y)


def test_relu_y_cuda() -> None:
    if not walnut.cuda.is_available():
        pass

    walnut_x, torch_x = get_vals(SHAPE, device="cuda")
    module = walnut.nn.layers.ReLU()
    module.training = True
    walnut_y = module(walnut_x)
    torch_y = F.relu(torch_x)
    assert validate(walnut_y, torch_y)


def test_relu_dx_cpu() -> None:
    walnut_x, torch_x = get_vals(SHAPE)
    module = walnut.nn.layers.ReLU()
    module.training = True
    _ = module(walnut_x)
    torch_y = F.relu(torch_x)
    walnut_dy, torch_dy = get_vals(SHAPE, torch_grad=False)
    walnut_dx = module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    assert validate(walnut_dx, torch_x.grad)


def test_relu_dx_cuda() -> None:
    if not walnut.cuda.is_available():
        pass

    walnut_x, torch_x = get_vals(SHAPE, device="cuda")
    module = walnut.nn.layers.ReLU()
    module.training = True
    _ = module(walnut_x)
    torch_y = F.relu(torch_x)
    walnut_dy, torch_dy = get_vals(SHAPE, torch_grad=False, device="cuda")
    walnut_dx = module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    assert validate(walnut_dx, torch_x.grad)


# Tanh
def test_tanh_y_cpu() -> None:
    walnut_x, torch_x = get_vals(SHAPE)
    module = walnut.nn.layers.Tanh()
    module.training = True
    walnut_y = module(walnut_x)
    torch_y = F.tanh(torch_x)
    assert validate(walnut_y, torch_y)


def test_tanh_y_cuda() -> None:
    if not walnut.cuda.is_available():
        pass

    walnut_x, torch_x = get_vals(SHAPE, device="cuda")
    module = walnut.nn.layers.Tanh()
    module.training = True
    walnut_y = module(walnut_x)
    torch_y = F.tanh(torch_x)
    assert validate(walnut_y, torch_y)


def test_tanh_dx_cpu() -> None:
    walnut_x, torch_x = get_vals(SHAPE)
    module = walnut.nn.layers.Tanh()
    module.training = True
    _ = module(walnut_x)
    torch_y = F.tanh(torch_x)
    walnut_dy, torch_dy = get_vals(SHAPE, torch_grad=False)
    walnut_dx = module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    assert validate(walnut_dx, torch_x.grad)


def test_tanh_dx_cuda() -> None:
    if not walnut.cuda.is_available():
        pass

    walnut_x, torch_x = get_vals(SHAPE, device="cuda")
    module = walnut.nn.layers.Tanh()
    module.training = True
    _ = module(walnut_x)
    torch_y = F.tanh(torch_x)
    walnut_dy, torch_dy = get_vals(SHAPE, torch_grad=False, device="cuda")
    walnut_dx = module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    assert validate(walnut_dx, torch_x.grad)


# Sigmoid
def test_sigmoid_y_cpu() -> None:
    walnut_x, torch_x = get_vals(SHAPE)
    module = walnut.nn.layers.Sigmoid()
    module.training = True
    walnut_y = module(walnut_x)
    torch_y = F.sigmoid(torch_x)
    assert validate(walnut_y, torch_y)


def test_sigmoid_y_cuda() -> None:
    if not walnut.cuda.is_available():
        pass

    walnut_x, torch_x = get_vals(SHAPE, device="cuda")
    module = walnut.nn.layers.Sigmoid()
    module.training = True
    walnut_y = module(walnut_x)
    torch_y = F.sigmoid(torch_x)
    assert validate(walnut_y, torch_y)


def test_sigmoid_dx_cpu() -> None:
    walnut_x, torch_x = get_vals(SHAPE)
    module = walnut.nn.layers.Sigmoid()
    module.training = True
    _ = module(walnut_x)
    torch_y = F.sigmoid(torch_x)
    walnut_dy, torch_dy = get_vals(SHAPE, torch_grad=False)
    walnut_dx = module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)
    assert validate(walnut_dx, torch_x.grad)


def test_sigmoid_dx_cuda() -> None:
    if not walnut.cuda.is_available():
        pass

    walnut_x, torch_x = get_vals(SHAPE, device="cuda")
    module = walnut.nn.layers.Sigmoid()
    module.training = True
    _ = module(walnut_x)
    torch_y = F.sigmoid(torch_x)
    walnut_dy, torch_dy = get_vals(SHAPE, torch_grad=False, device="cuda")
    walnut_dx = module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)

    assert validate(walnut_dx, torch_x.grad)
