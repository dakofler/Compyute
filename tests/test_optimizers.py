"""Optimizers tests"""

import torch

from src.compyute.nn.optimizers import SGD, Adam, AdamW, NAdam
from tests.test_utils import get_random_floats, get_random_params, is_equal

SHAPE = (10, 20)
ITER = 10
EPS = 1e-8
BETA1 = 0.9
BETA2 = 0.999
WDECAY = 0.1
MOMENTUM = 0.1
MOMENTUM_DECAY = 4e-3


def test_sgd() -> None:
    """Test for the stochastic gradient descent optimizer."""
    lr = 1e-2

    # init parameters
    compyute_x, torch_x = get_random_params(SHAPE)
    compyute_dx, torch_dx = get_random_floats(SHAPE, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    # init compyute optimizer
    compyute_optim = SGD([compyute_x], lr=lr)

    # init torch optimizer
    torch_optim = torch.optim.SGD([torch_x], lr=lr)

    # forward
    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    assert is_equal(compyute_x, torch_x)
    assert is_equal(compyute_x.grad, torch_x.grad)


def test_sgd_m() -> None:
    """Test for the stochastic gradient descent optimizer using momentum."""
    lr = 1e-2

    # init parameters
    compyute_x, torch_x = get_random_params(SHAPE)
    compyute_dx, torch_dx = get_random_floats(SHAPE, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    # init compyute optimizer
    compyute_optim = SGD([compyute_x], lr=lr, momentum=MOMENTUM)

    # init torch optimizer
    torch_optim = torch.optim.SGD([torch_x], lr=lr, momentum=MOMENTUM)

    # forward
    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    assert is_equal(compyute_x, torch_x)
    assert is_equal(compyute_x.grad, torch_x.grad)


def test_sgd_m_nesterov() -> None:
    """Test for the stochastic gradient descent optimizer using nesterov momentum."""
    lr = 1e-2
    nesterov = True

    # init parameters
    compyute_x, torch_x = get_random_params(SHAPE)
    compyute_dx, torch_dx = get_random_floats(SHAPE, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    # init compyute optimizer
    compyute_optim = SGD([compyute_x], lr=lr, momentum=MOMENTUM, nesterov=nesterov)

    # init torch optimizer
    torch_optim = torch.optim.SGD([torch_x], lr=lr, momentum=MOMENTUM, nesterov=nesterov)

    # forward
    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    assert is_equal(compyute_x, torch_x)
    assert is_equal(compyute_x.grad, torch_x.grad)


def test_sgd_m_wdecay() -> None:
    """Test for the stochastic gradient descent optimizer using weight decay."""
    lr = 1e-2

    # init parameters
    compyute_x, torch_x = get_random_params(SHAPE)
    compyute_dx, torch_dx = get_random_floats(SHAPE, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    # init compyute optimizer
    compyute_optim = SGD([compyute_x], lr=lr, momentum=MOMENTUM, weight_decay=WDECAY)

    # init torch optimizer
    torch_optim = torch.optim.SGD([torch_x], lr=lr, momentum=MOMENTUM, weight_decay=WDECAY)

    # forward
    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    assert is_equal(compyute_x, torch_x)
    assert is_equal(compyute_x.grad, torch_x.grad)


def test_adam() -> None:
    """Test for the adam optimizer."""
    lr = 1e-3

    # init parameters
    compyute_x, torch_x = get_random_params(SHAPE)
    compyute_dx, torch_dx = get_random_floats(SHAPE, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    # init compyute optimizer
    compyute_optim = Adam([compyute_x], lr=lr, beta1=BETA1, beta2=BETA2, eps=EPS)

    # init torch optimizer
    torch_optim = torch.optim.Adam([torch_x], lr=lr, betas=(BETA1, BETA2), eps=EPS)

    # forward
    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    assert is_equal(compyute_x, torch_x)
    assert is_equal(compyute_x.grad, torch_x.grad)


def test_adam_wdecay() -> None:
    """Test for the adam optimizer using weight decay."""
    lr = 1e-3

    # init parameters
    compyute_x, torch_x = get_random_params(SHAPE)
    compyute_dx, torch_dx = get_random_floats(SHAPE, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    # init compyute optimizer
    compyute_optim = Adam(
        [compyute_x], lr=lr, beta1=BETA1, beta2=BETA2, eps=EPS, weight_decay=WDECAY
    )

    # init torch optimizer
    torch_optim = torch.optim.Adam(
        [torch_x], lr=lr, betas=(BETA1, BETA2), eps=EPS, weight_decay=WDECAY
    )

    # forward
    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    assert is_equal(compyute_x, torch_x)
    assert is_equal(compyute_x.grad, torch_x.grad)


def test_adamw() -> None:
    """Test for the adamW optimizer."""
    lr = 1e-3

    # init parameters
    compyute_x, torch_x = get_random_params(SHAPE)
    compyute_dx, torch_dx = get_random_floats(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    # init compyute optimizer
    compyute_optim = AdamW([compyute_x], lr=lr, beta1=BETA1, beta2=BETA2, eps=EPS)

    # init torch optimizer
    torch_optim = torch.optim.AdamW([torch_x], lr=lr, betas=(BETA1, BETA2), eps=EPS)

    # forward
    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    assert is_equal(compyute_x, torch_x)
    assert is_equal(compyute_x.grad, torch_x.grad)


def test_nadam() -> None:
    """Test for the nadam optimizer."""
    lr = 2e-3

    # init parameters
    compyute_x, torch_x = get_random_params(SHAPE)
    compyute_dx, torch_dx = get_random_floats(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    # init compyute optimizer
    compyute_optim = NAdam(
        [compyute_x],
        lr=lr,
        beta1=BETA1,
        beta2=BETA2,
        eps=EPS,
        momentum_decay=MOMENTUM_DECAY,
    )

    # init torch optimizer
    torch_optim = torch.optim.NAdam(
        [torch_x],
        lr=lr,
        betas=(BETA1, BETA2),
        eps=EPS,
        momentum_decay=MOMENTUM_DECAY,
    )

    # forward
    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    assert is_equal(compyute_x, torch_x)
    assert is_equal(compyute_x.grad, torch_x.grad)
