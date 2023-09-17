"""Losses tests"""

import torch
import walnut
from tests.test_utils import get_vals, validate


SHAPE = (10, 10, 10)


# MSE
def test_mse_cpu() -> None:
    results = []

    # forward
    walnut_x, torch_x = get_vals(SHAPE)
    walnut_t, torch_t = get_vals(SHAPE)

    walnut_loss = walnut.nn.losses.MSE()
    walnut_y = walnut_loss(walnut_x, walnut_t)

    torch_loss = torch.nn.MSELoss()
    torch_y = torch_loss(torch_x, torch_t)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dx = walnut_loss.backward()
    torch_y.backward()
    results.append(validate(walnut_dx, torch_x.grad))

    assert all(results)


def test_mse_cuda() -> None:
    if not walnut.cuda.is_available():
        pass
    results = []

    # forward
    walnut_x, torch_x = get_vals(SHAPE, device="cuda")
    walnut_t, torch_t = get_vals(SHAPE, device="cuda")

    walnut_loss = walnut.nn.losses.MSE()
    walnut_y = walnut_loss(walnut_x, walnut_t)

    torch_loss = torch.nn.MSELoss()
    torch_y = torch_loss(torch_x, torch_t)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dx = walnut_loss.backward()
    torch_y.backward()
    results.append(validate(walnut_dx, torch_x.grad))

    assert all(results)


# Crossentropy
def test_crossentropy_cpu() -> None:
    results = []

    # forward
    walnut_x, _ = get_vals(SHAPE)
    walnut_t = walnut.randint(SHAPE[:-1], 0, walnut_x.shape[-1])
    torch_x = torch.from_numpy(walnut_x.moveaxis(-1, -2).data)
    torch_x.requires_grad = True
    torch_t = torch.from_numpy(walnut_t.data).long()

    walnut_loss = walnut.nn.losses.Crossentropy()
    walnut_y = walnut_loss(walnut_x, walnut_t)

    torch_loss = torch.nn.CrossEntropyLoss()
    torch_y = torch_loss(torch_x, torch_t)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dx = walnut_loss.backward()
    torch_y.backward()
    results.append(validate(walnut_dx, torch_x.grad.moveaxis(-1, -2)))

    assert all(results)


# Crossentropy
def test_crossentropy_cuda() -> None:
    if not walnut.cuda.is_available():
        pass
    results = []

    # forward
    walnut_x, _ = get_vals(SHAPE)
    walnut_t = walnut.randint(SHAPE[:-1], 0, walnut_x.shape[-1])
    torch_x = torch.from_numpy(walnut_x.moveaxis(-1, -2).data)
    torch_x.requires_grad = True
    torch_t = torch.from_numpy(walnut_t.data).long()
    walnut_x.to_device("cuda")
    walnut_t.to_device("cuda")

    walnut_loss = walnut.nn.losses.Crossentropy()
    walnut_y = walnut_loss(walnut_x, walnut_t)

    torch_loss = torch.nn.CrossEntropyLoss()
    torch_y = torch_loss(torch_x, torch_t)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dx = walnut_loss.backward()
    torch_y.backward()
    results.append(validate(walnut_dx, torch_x.grad.moveaxis(-1, -2)))

    assert all(results)
