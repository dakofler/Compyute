"""Optimizers tests"""

import torch
from compyute.nn.trainer.optimizers import SGD, Adam, AdamW, NAdam
from tests.test_utils import get_vals_float, get_params, validate


SHAPE = (10, 20)
ITER = 10


def test_sgd() -> None:
    """Test for the stochastic gradient descent optimizer."""
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    compyute_optim = SGD(lr=1e-2)
    compyute_optim.parameters = [compyute_x]

    torch_optim = torch.optim.SGD([torch_x], lr=1e-2)

    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    results.append(validate(compyute_x, torch_x))
    results.append(validate(compyute_x.grad, torch_x.grad))

    assert all(results)


def test_sgd_m() -> None:
    """Test for the stochastic gradient descent optimizer using momentum."""
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    compyute_optim = SGD(lr=1e-2, momentum=0.1)
    compyute_optim.parameters = [compyute_x]

    torch_optim = torch.optim.SGD([torch_x], lr=1e-2, momentum=0.1)

    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    results.append(validate(compyute_x, torch_x))
    results.append(validate(compyute_x.grad, torch_x.grad))

    assert all(results)


def test_sgd_m_nesterov() -> None:
    """Test for the stochastic gradient descent optimizer using nesterov momentum."""
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    compyute_optim = SGD(lr=1e-2, momentum=0.1, nesterov=True)
    compyute_optim.parameters = [compyute_x]

    torch_optim = torch.optim.SGD([torch_x], lr=1e-2, momentum=0.1, nesterov=True)

    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    results.append(validate(compyute_x, torch_x))
    results.append(validate(compyute_x.grad, torch_x.grad))

    assert all(results)


def test_sgd_m_wdecay() -> None:
    """Test for the stochastic gradient descent optimizer using weight decay."""
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    compyute_optim = SGD(lr=1e-2, momentum=0.1, weight_decay=0.1)
    compyute_optim.parameters = [compyute_x]
    torch_optim = torch.optim.SGD([torch_x], lr=1e-2, momentum=0.1, weight_decay=0.1)

    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    results.append(validate(compyute_x, torch_x))
    results.append(validate(compyute_x.grad, torch_x.grad))

    assert all(results)


def test_adam() -> None:
    """Test for the adam optimizer."""
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    compyute_optim = Adam(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)
    compyute_optim.parameters = [compyute_x]

    torch_optim = torch.optim.Adam([torch_x], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    results.append(validate(compyute_x, torch_x))
    results.append(validate(compyute_x.grad, torch_x.grad))

    assert all(results)


def test_adam_wdecay() -> None:
    """Test for the adam optimizer using weight decay."""
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    compyute_optim = Adam(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1)
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
    """Test for the adamW optimizer."""
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    compyute_optim = AdamW(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)
    compyute_optim.parameters = [compyute_x]

    torch_optim = torch.optim.AdamW([torch_x], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    results.append(validate(compyute_x, torch_x))
    results.append(validate(compyute_x.grad, torch_x.grad))

    assert all(results)


def test_adamw_wdecay() -> None:
    """Test for the adamW optimizer using weight decay."""
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    compyute_optim = AdamW(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1)
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


def test_nadam() -> None:
    """Test for the nadam optimizer."""
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals_float(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    compyute_optim = NAdam(
        lr=2e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0, momentum_decay=4e-3
    )
    compyute_optim.parameters = [compyute_x]

    torch_optim = torch.optim.NAdam(
        [torch_x],
        lr=2e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        momentum_decay=4e-3,
    )

    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    results.append(validate(compyute_x, torch_x))
    results.append(validate(compyute_x.grad, torch_x.grad))

    assert all(results)
