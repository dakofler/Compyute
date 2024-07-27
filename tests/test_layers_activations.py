"""Activation module tests"""

import torch.nn.functional as F

from compyute.nn import GELU, LeakyReLU, ReLU, Sigmoid, Tanh
from tests.test_utils import get_random_floats, is_equal

SHAPE = (10, 20, 30)


def test_relu() -> None:
    """Test for the relu layer."""
    # init parameters
    compyute_x, torch_x = get_random_floats(SHAPE)

    # init compyute module
    compyute_module = ReLU(training=True)

    # forward
    compyute_y = compyute_module(compyute_x)
    torch_y = F.relu(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(SHAPE, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)


def test_leaky_relu() -> None:
    """Test for the leaky relu layer."""

    # init parameters
    compyute_x, torch_x = get_random_floats(SHAPE)

    # init compyute module
    compyute_module = LeakyReLU(alpha=0.01, training=True)

    # forward
    compyute_y = compyute_module(compyute_x)
    torch_y = F.leaky_relu(torch_x, negative_slope=0.01)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(SHAPE, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)


def test_gelu() -> None:
    """Test for the gelu layer."""

    # init parameters
    compyute_x, torch_x = get_random_floats(SHAPE)

    # init compyute module
    compyute_module = GELU(training=True)

    # forward
    compyute_y = compyute_module(compyute_x)
    torch_y = F.gelu(torch_x, approximate="tanh")
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(SHAPE, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)


def test_tanh() -> None:
    """Test for the tanh layer."""

    # init parameters
    compyute_x, torch_x = get_random_floats(SHAPE)

    # init compyute module
    compyute_module = Tanh(training=True)

    # forward
    compyute_y = compyute_module(compyute_x)
    torch_y = F.tanh(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(SHAPE, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)


def test_sigmoid() -> None:
    """Test for the sigmoid layer."""

    # init parameters
    compyute_x, torch_x = get_random_floats(SHAPE)

    # init compyute module
    compyute_module = Sigmoid(training=True)

    # forward
    compyute_y = compyute_module(compyute_x)
    torch_y = F.sigmoid(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(SHAPE, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_dx, torch_x.grad)
