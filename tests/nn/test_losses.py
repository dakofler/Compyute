"""Loss function tests"""

import pytest
import torch

from compyute.nn import BinaryCrossEntropy, CrossEntropy, MeanSquaredError
from tests.utils import get_random_floats, get_random_integers, is_close

testdata = [(8, 16), (8, 16, 32), (8, 16, 32, 64)]


@pytest.mark.parametrize("shape", testdata)
def test_mse(shape) -> None:
    """Test for the mean squared error loss using 2d inputs."""

    # init compyute loss
    compyute_loss = MeanSquaredError()

    # init torch loss
    torch_loss = torch.nn.MSELoss()

    # forward
    compyute_x, torch_x = get_random_floats(shape)
    compyute_t, torch_t = get_random_floats(shape)
    compyute_y = compyute_loss(compyute_x, compyute_t)
    torch_y = torch_loss(torch_x, torch_t)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    assert is_close(compyute_dx, torch_x.grad)


@pytest.mark.parametrize("shape", testdata)
def test_cross_entropy(shape) -> None:
    """Test for the cross entropy loss using 2d inputs."""

    # init compyute loss
    compyute_loss = CrossEntropy()

    # init torch loss
    torch_loss = torch.nn.CrossEntropyLoss()

    # forward
    compyute_x, torch_x = get_random_floats(shape)
    torch_x_ma = torch.moveaxis(torch_x, -1, 1) if len(shape) > 2 else torch_x
    compyute_t, torch_t = get_random_integers(shape[:-1], high=shape[-1])
    compyute_y = compyute_loss(compyute_x, compyute_t)
    torch_y = torch_loss(torch_x_ma, torch_t)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    assert is_close(compyute_dx, torch_x.grad)


@pytest.mark.parametrize("shape", testdata)
def test_binary_cross_entropy(shape) -> None:
    """Test for the binary cross entropy loss using 2d inputs."""

    # init compyute loss
    compyute_loss = BinaryCrossEntropy()

    # init torch loss
    torch_loss = torch.nn.BCELoss()

    # forward
    compyute_x, torch_x = get_random_floats(shape, low=0)
    compyute_t, torch_t = get_random_floats(shape, low=0)
    compyute_y = compyute_loss(compyute_x, compyute_t)
    torch_y = torch_loss(torch_x, torch_t)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    assert is_close(compyute_dx, torch_x.grad)
