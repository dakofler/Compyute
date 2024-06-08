"""Losses tests"""

import torch

from src.compyute.nn.trainer.losses import BinaryCrossEntropy, CrossEntropy, MeanSquaredError
from tests.test_utils import get_vals_float, get_vals_int, validate

SHAPE2D = (20, 5)
SHAPE3D = (20, 10, 5)


def test_mse_2d() -> None:
    """Test for the mean squared error loss using 2d inputs."""
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE2D)
    compyute_t, torch_t = get_vals_float(SHAPE2D)

    compyute_loss = MeanSquaredError()
    compyute_y = compyute_loss(compyute_x, compyute_t)

    torch_loss = torch.nn.MSELoss()
    torch_y = torch_loss(torch_x, torch_t)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_mse_nd() -> None:
    """Test for the mean squared error loss using nd inputs."""
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE3D)
    compyute_t, torch_t = get_vals_float(SHAPE3D)

    compyute_loss = MeanSquaredError()
    compyute_y = compyute_loss(compyute_x, compyute_t)

    torch_loss = torch.nn.MSELoss()
    torch_y = torch_loss(torch_x, torch_t)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_cross_entropy_2d() -> None:
    """Test for the cross entropy loss using 2d inputs."""
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE2D)
    compyute_t, torch_t = get_vals_int((SHAPE2D[0],), high=SHAPE2D[1])

    compyute_loss = CrossEntropy()
    compyute_y = compyute_loss(compyute_x, compyute_t)

    torch_loss = torch.nn.CrossEntropyLoss()
    torch_y = torch_loss(torch_x, torch_t)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_cross_entropy_nd() -> None:
    """Test for the cross entropy loss using nd inputs."""
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE3D)
    torch_x_ma = torch.moveaxis(torch_x, -2, -1)
    compyute_t, torch_t = get_vals_int(SHAPE3D[:2], high=SHAPE3D[2])

    compyute_loss = CrossEntropy()
    compyute_y = compyute_loss(compyute_x, compyute_t)

    torch_loss = torch.nn.CrossEntropyLoss()
    torch_y = torch_loss(torch_x_ma, torch_t)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_binary_cross_entropy_2d() -> None:
    """Test for the binary cross entropy loss using 2d inputs."""
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE2D, low=0)
    compyute_t, torch_t = get_vals_float(SHAPE2D, low=0)

    compyute_loss = BinaryCrossEntropy()
    compyute_y = compyute_loss(compyute_x, compyute_t)

    torch_loss = torch.nn.BCELoss()
    torch_y = torch_loss(torch_x, torch_t)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_binary_cross_entropy_nd() -> None:
    """Test for the binary cross entropy loss using nd inputs."""
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE3D, low=0)
    compyute_t, torch_t = get_vals_float(SHAPE3D, low=0)

    compyute_loss = BinaryCrossEntropy()
    compyute_y = compyute_loss(compyute_x, compyute_t)

    torch_loss = torch.nn.BCELoss()
    torch_y = torch_loss(torch_x, torch_t)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)
