"""Losses tests"""

import torch
from compyute.nn.trainer.losses import MSE, BinaryCrossentropy, Crossentropy
from tests.test_utils import get_vals_float, validate, get_vals_int


SHAPE = (20, 5)


def test_mse() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE)
    compyute_t, torch_t = get_vals_float(SHAPE)

    compyute_loss = MSE()
    compyute_y = compyute_loss(compyute_x, compyute_t)

    torch_loss = torch.nn.MSELoss()
    torch_y = torch_loss(torch_x, torch_t)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_crossentropy() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE)
    compyute_t, torch_t = get_vals_int((SHAPE[0],), high=SHAPE[1])

    compyute_loss = Crossentropy()
    compyute_y = compyute_loss(compyute_x, compyute_t)

    torch_loss = torch.nn.CrossEntropyLoss()
    torch_y = torch_loss(torch_x, torch_t)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_binary_crossentropy() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_vals_float(SHAPE, low=0)
    compyute_t, torch_t = get_vals_float(SHAPE, low=0)

    compyute_loss = BinaryCrossentropy()
    compyute_y = compyute_loss(compyute_x, compyute_t)

    torch_loss = torch.nn.BCELoss()
    torch_y = torch_loss(torch_x, torch_t)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)
