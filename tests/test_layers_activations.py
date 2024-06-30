"""Activation layer tests"""

import torch.nn.functional as F

from src.compyute.nn import GELU, LeakyReLU, ReLU, Sigmoid, Tanh
from tests.test_utils import get_vals_float, validate

SHAPE = (10, 20, 30)


def test_relu() -> None:
    """Test for the relu layer."""
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE)
    module = ReLU()
    module.set_training(True)
    compyute_y = module(compyute_x)
    torch_y = F.relu(torch_x)
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(SHAPE, torch_grad=False)
    compyute_dx = module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_leaky_relu() -> None:
    """Test for the leaky relu layer."""
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE)
    module = LeakyReLU(alpha=0.01)
    module.set_training(True)
    compyute_y = module(compyute_x)
    torch_y = F.leaky_relu(torch_x, negative_slope=0.01)
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(SHAPE, torch_grad=False)
    compyute_dx = module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_gelu() -> None:
    """Test for the gelu layer."""
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE)
    module = GELU()
    module.set_training(True)
    compyute_y = module(compyute_x)
    torch_y = F.gelu(torch_x, approximate="tanh")
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(SHAPE, torch_grad=False)
    compyute_dx = module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_tanh() -> None:
    """Test for the tanh layer."""
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE)
    module = Tanh()
    module.set_training(True)
    compyute_y = module(compyute_x)
    torch_y = F.tanh(torch_x)
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(SHAPE, torch_grad=False)
    compyute_dx = module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_sigmoid() -> None:
    """Test for the sigmoid layer."""
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE)
    module = Sigmoid()
    module.set_training(True)
    compyute_y = module(compyute_x)
    torch_y = F.sigmoid(torch_x)
    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(SHAPE, torch_grad=False)
    compyute_dx = module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)
