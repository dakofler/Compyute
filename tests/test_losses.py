"""Loss function tests"""

import torch

from compyute.nn import BinaryCrossEntropy, CrossEntropy, MeanSquaredError
from tests.test_utils import get_random_floats, get_random_integers, is_equal

SHAPE2D = (20, 5)
SHAPE3D = (20, 10, 5)


def test_mse_2d() -> None:
    """Test for the mean squared error loss using 2d inputs."""
    shape_x = SHAPE2D

    # init compyute loss
    compyute_loss = MeanSquaredError()

    # init torch loss
    torch_loss = torch.nn.MSELoss()

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_t, torch_t = get_random_floats(shape_x)
    compyute_y = compyute_loss(compyute_x, compyute_t)
    torch_y = torch_loss(torch_x, torch_t)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    assert is_equal(compyute_dx, torch_x.grad)


def test_mse_nd() -> None:
    """Test for the mean squared error loss using nd inputs."""
    shape_x = SHAPE3D

    # init compyute loss
    compyute_loss = MeanSquaredError()

    # init torch loss
    torch_loss = torch.nn.MSELoss()

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_t, torch_t = get_random_floats(shape_x)
    compyute_y = compyute_loss(compyute_x, compyute_t)
    torch_y = torch_loss(torch_x, torch_t)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    assert is_equal(compyute_dx, torch_x.grad)


def test_cross_entropy_2d() -> None:
    """Test for the cross entropy loss using 2d inputs."""
    shape_x = SHAPE2D

    # init compyute loss
    compyute_loss = CrossEntropy()

    # init torch loss
    torch_loss = torch.nn.CrossEntropyLoss()

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_t, torch_t = get_random_integers((shape_x[0],), high=shape_x[1])
    compyute_y = compyute_loss(compyute_x, compyute_t)
    torch_y = torch_loss(torch_x, torch_t)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    assert is_equal(compyute_dx, torch_x.grad)


def test_cross_entropy_nd() -> None:
    """Test for the cross entropy loss using nd inputs."""
    shape_x = SHAPE3D

    # init compyute loss
    compyute_loss = CrossEntropy()

    # init torch loss
    torch_loss = torch.nn.CrossEntropyLoss()

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    torch_x_ma = torch.moveaxis(torch_x, -2, -1)
    compyute_t, torch_t = get_random_integers(shape_x[:2], high=shape_x[2])
    compyute_y = compyute_loss(compyute_x, compyute_t)
    torch_y = torch_loss(torch_x_ma, torch_t)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    assert is_equal(compyute_dx, torch_x.grad)


def test_binary_cross_entropy_2d() -> None:
    """Test for the binary cross entropy loss using 2d inputs."""
    shape_x = SHAPE2D

    # init compyute loss
    compyute_loss = BinaryCrossEntropy()

    # init torch loss
    torch_loss = torch.nn.BCELoss()

    # forward
    compyute_x, torch_x = get_random_floats(shape_x, low=0)
    compyute_t, torch_t = get_random_floats(shape_x, low=0)
    compyute_y = compyute_loss(compyute_x, compyute_t)
    torch_y = torch_loss(torch_x, torch_t)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    assert is_equal(compyute_dx, torch_x.grad)


def test_binary_cross_entropy_nd() -> None:
    """Test for the binary cross entropy loss using nd inputs."""
    shape_x = SHAPE3D

    # init compyute loss
    compyute_loss = BinaryCrossEntropy()

    # init torch loss
    torch_loss = torch.nn.BCELoss()

    # forward
    compyute_x, torch_x = get_random_floats(shape_x, low=0)
    compyute_t, torch_t = get_random_floats(shape_x, low=0)
    compyute_y = compyute_loss(compyute_x, compyute_t)
    torch_y = torch_loss(torch_x, torch_t)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    assert is_equal(compyute_dx, torch_x.grad)
