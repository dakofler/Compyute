"""Activation module tests"""

import pytest
import torch.nn.functional as F

from compyute.nn import GELU, FastGELU, LeakyReLU, ReLU, Sigmoid, SiLU, Softmax, Tanh
from tests.utils import get_random_floats, is_close

testdata = [(8, 16), (8, 16, 32), (8, 16, 32, 64)]


@pytest.mark.parametrize("shape", testdata)
def test_relu(shape) -> None:
    """Test for the relu layer."""
    # init parameters
    compyute_x, torch_x = get_random_floats(shape)

    # init compyute module
    compyute_module = ReLU()

    # forward
    compyute_y = compyute_module(compyute_x)
    torch_y = F.relu(torch_x)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)


@pytest.mark.parametrize("shape", testdata)
@pytest.mark.parametrize("alpha", [0.01, 0.02, 0.1])
def test_leaky_relu(shape, alpha) -> None:
    """Test for the leaky relu layer."""

    # init parameters
    compyute_x, torch_x = get_random_floats(shape)

    # init compyute module
    compyute_module = LeakyReLU(alpha=alpha)

    # forward
    compyute_y = compyute_module(compyute_x)
    torch_y = F.leaky_relu(torch_x, negative_slope=alpha)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)


@pytest.mark.parametrize("shape", testdata)
def test_gelu(shape) -> None:
    """Test for the gelu layer."""

    # init parameters
    compyute_x, torch_x = get_random_floats(shape)

    # init compyute module
    compyute_module = GELU()

    # forward
    compyute_y = compyute_module(compyute_x)
    torch_y = F.gelu(torch_x, approximate="tanh")
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)


@pytest.mark.parametrize("shape", testdata)
def test_fast_gelu(shape) -> None:
    """Test for the fast gelu layer."""

    # init parameters
    compyute_x, torch_x = get_random_floats(shape)

    # init compyute module
    compyute_module = FastGELU()

    # forward
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_x * F.sigmoid(1.702 * torch_x)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)


@pytest.mark.parametrize("shape", testdata)
def test_tanh(shape) -> None:
    """Test for the tanh layer."""

    # init parameters
    compyute_x, torch_x = get_random_floats(shape)

    # init compyute module
    compyute_module = Tanh()

    # forward
    compyute_y = compyute_module(compyute_x)
    torch_y = F.tanh(torch_x)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)


@pytest.mark.parametrize("shape", testdata)
def test_sigmoid(shape) -> None:
    """Test for the sigmoid layer."""

    # init parameters
    compyute_x, torch_x = get_random_floats(shape)

    # init compyute module
    compyute_module = Sigmoid()

    # forward
    compyute_y = compyute_module(compyute_x)
    torch_y = F.sigmoid(torch_x)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)


@pytest.mark.parametrize("shape", testdata)
def test_silu(shape) -> None:
    """Test for the silu layer."""

    # init parameters
    compyute_x, torch_x = get_random_floats(shape)

    # init compyute module
    compyute_module = SiLU()

    # forward
    compyute_y = compyute_module(compyute_x)
    torch_y = F.silu(torch_x)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)


@pytest.mark.parametrize("shape", testdata)
def test_softmax(shape) -> None:
    """Test for the softmax layer."""

    # init parameters
    compyute_x, torch_x = get_random_floats(shape)

    # init compyute module
    compyute_module = Softmax()

    # forward
    compyute_y = compyute_module(compyute_x)
    torch_y = F.softmax(torch_x, dim=-1)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)
