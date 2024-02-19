"""Losses tests"""

import torch
import compyute
from tests.test_utils import get_vals, validate


SHAPE = (10, 10, 10)


# MSE
def test_mse_cpu() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_vals(SHAPE)
    compyute_t, torch_t = get_vals(SHAPE)

    compyute_loss = compyute.nn.losses.MSE()
    compyute_y = compyute_loss(compyute_x, compyute_t)

    torch_loss = torch.nn.MSELoss()
    torch_y = torch_loss(torch_x, torch_t)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_mse_cuda() -> None:
    if not compyute.cuda.is_available():
        pass
    results = []

    # forward
    compyute_x, torch_x = get_vals(SHAPE, device="cuda")
    compyute_t, torch_t = get_vals(SHAPE, device="cuda")

    compyute_loss = compyute.nn.losses.MSE()
    compyute_y = compyute_loss(compyute_x, compyute_t)

    torch_loss = torch.nn.MSELoss()
    torch_y = torch_loss(torch_x, torch_t)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


# Crossentropy
def test_crossentropy_cpu() -> None:
    results = []

    # forward
    compyute_x, _ = get_vals(SHAPE)
    compyute_t = compyute.randint(SHAPE[:-1], 0, compyute_x.shape[-1])
    torch_x = torch.from_numpy(compyute_x.moveaxis(-1, -2).data)
    torch_x.requires_grad = True
    torch_t = torch.from_numpy(compyute_t.data).long()

    compyute_loss = compyute.nn.losses.Crossentropy()
    compyute_y = compyute_loss(compyute_x, compyute_t)

    torch_loss = torch.nn.CrossEntropyLoss()
    torch_y = torch_loss(torch_x, torch_t)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    results.append(validate(compyute_dx, torch_x.grad.moveaxis(-1, -2)))

    assert all(results)


# Crossentropy
def test_crossentropy_cuda() -> None:
    if not compyute.cuda.is_available():
        pass
    results = []

    # forward
    compyute_x, _ = get_vals(SHAPE)
    compyute_t = compyute.randint(SHAPE[:-1], 0, compyute_x.shape[-1])
    torch_x = torch.from_numpy(compyute_x.moveaxis(-1, -2).data)
    torch_x.requires_grad = True
    torch_t = torch.from_numpy(compyute_t.data).long()
    compyute_x.to_device("cuda")
    compyute_t.to_device("cuda")

    compyute_loss = compyute.nn.losses.Crossentropy()
    compyute_y = compyute_loss(compyute_x, compyute_t)

    torch_loss = torch.nn.CrossEntropyLoss()
    torch_y = torch_loss(torch_x, torch_t)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    results.append(validate(compyute_dx, torch_x.grad.moveaxis(-1, -2)))

    assert all(results)
