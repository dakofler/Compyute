"""Optimizers tests"""

import torch
import compyute
from tests.test_utils import get_vals_float, get_params, validate


SHAPE = (10, 20)
ITER = 10


def test_sgd() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx.data
    torch_x.grad = torch_dx

    compyute_optim = compyute.nn.optimizers.SGD(lr=1e-2)
    compyute_optim.parameters = [compyute_x]

    torch_optim = torch.optim.SGD([torch_x], lr=1e-2)

    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    results.append(validate(compyute_x, torch_x))
    results.append(validate(compyute_x.grad, torch_x.grad))

    assert all(results)


def test_sgd_m() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx.data
    torch_x.grad = torch_dx

    compyute_optim = compyute.nn.optimizers.SGD(lr=1e-2, m=0.1)
    compyute_optim.parameters = [compyute_x]

    torch_optim = torch.optim.SGD([torch_x], lr=1e-2, momentum=0.1)

    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    results.append(validate(compyute_x, torch_x))
    results.append(validate(compyute_x.grad, torch_x.grad))

    assert all(results)


def test_sgd_m_nesterov() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx.data
    torch_x.grad = torch_dx

    compyute_optim = compyute.nn.optimizers.SGD(lr=1e-2, m=0.1, nesterov=True)
    compyute_optim.parameters = [compyute_x]

    torch_optim = torch.optim.SGD([torch_x], lr=1e-2, momentum=0.1, nesterov=True)

    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    results.append(validate(compyute_x, torch_x, tol=1e-4))
    results.append(validate(compyute_x.grad, torch_x.grad, tol=1e-4))

    assert all(results)


def test_sgd_m_wdecay() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx.data
    torch_x.grad = torch_dx

    compyute_optim = compyute.nn.optimizers.SGD(lr=1e-2, m=0.1, weight_decay=0.1)
    compyute_optim.parameters = [compyute_x]
    torch_optim = torch.optim.SGD([torch_x], lr=1e-2, momentum=0.1, weight_decay=0.1)

    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    results.append(validate(compyute_x, torch_x))
    results.append(validate(compyute_x.grad, torch_x.grad))

    assert all(results)


def test_adam() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx.data
    torch_x.grad = torch_dx

    compyute_optim = compyute.nn.optimizers.Adam(
        lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8
    )
    compyute_optim.parameters = [compyute_x]

    torch_optim = torch.optim.Adam([torch_x], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    results.append(validate(compyute_x, torch_x))
    results.append(validate(compyute_x.grad, torch_x.grad))

    assert all(results)


def test_adam_wdecay() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx.data
    torch_x.grad = torch_dx

    compyute_optim = compyute.nn.optimizers.Adam(
        lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1
    )
    compyute_optim.parameters = [compyute_x]

    torch_optim = torch.optim.Adam(
        [torch_x], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1
    )

    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    results.append(validate(compyute_x, torch_x))
    results.append(validate(compyute_x.grad, torch_x.grad))
    assert all(results)


def test_adamw() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx.data
    torch_x.grad = torch_dx

    compyute_optim = compyute.nn.optimizers.AdamW(
        lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8
    )
    compyute_optim.parameters = [compyute_x]

    torch_optim = torch.optim.AdamW([torch_x], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    results.append(validate(compyute_x, torch_x))
    results.append(validate(compyute_x.grad, torch_x.grad))

    assert all(results)


def test_adamw_wdecay() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx.data
    torch_x.grad = torch_dx

    compyute_optim = compyute.nn.optimizers.AdamW(
        lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1
    )
    compyute_optim.parameters = [compyute_x]

    torch_optim = torch.optim.AdamW(
        [torch_x], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1
    )

    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    results.append(validate(compyute_x, torch_x))
    results.append(validate(compyute_x.grad, torch_x.grad))

    assert all(results)
