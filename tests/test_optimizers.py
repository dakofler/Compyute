"""Optimizer tests"""

import pytest
import torch

from compyute.nn.optimizers import SGD, Adam, AdamW, NAdam
from tests.test_utils import get_random_floats, get_random_params, is_equal

iters = 5
shape_testdata = [(8, 16), (8, 16, 32)]
lr_testdata = [1e-2, 1e-3, 3e-4]
eps_testdata = [1e-8]
beta1_testdata = [0.9]
beta2_testdata = [0.999, 0.95]
weight_decay_testdata = [0.0, 0.01, 0.001]


@pytest.mark.parametrize("shape", shape_testdata)
@pytest.mark.parametrize("lr", lr_testdata)
@pytest.mark.parametrize("m", [0.0, 0.1])
@pytest.mark.parametrize("weight_decay", weight_decay_testdata)
def test_sgd(shape, lr, m, weight_decay) -> None:
    """Test for the stochastic gradient descent optimizer."""

    # init parameters
    compyute_x, torch_x = get_random_params(shape)
    compyute_dx, torch_dx = get_random_floats(shape, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    # init compyute optimizer
    compyute_optim = SGD([compyute_x], lr, m, weight_decay)

    # init torch optimizer
    torch_optim = torch.optim.SGD([torch_x], lr, m, weight_decay=weight_decay)

    # forward
    for _ in range(iters):
        compyute_optim.step()
        torch_optim.step()

    assert is_equal(compyute_x, torch_x, tol=1e-4)
    assert is_equal(compyute_x.grad, torch_x.grad)


@pytest.mark.parametrize("shape", shape_testdata)
@pytest.mark.parametrize("lr", lr_testdata)
@pytest.mark.parametrize("m", [0.1, 0.2])
@pytest.mark.parametrize("weight_decay", weight_decay_testdata)
def test_sgd_nesterov(shape, lr, m, weight_decay) -> None:
    """Test for the stochastic gradient descent optimizer."""

    # init parameters
    compyute_x, torch_x = get_random_params(shape)
    compyute_dx, torch_dx = get_random_floats(shape, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    # init compyute optimizer
    compyute_optim = SGD([compyute_x], lr, m, True, weight_decay)

    # init torch optimizer
    torch_optim = torch.optim.SGD(
        [torch_x], lr, m, weight_decay=weight_decay, nesterov=True
    )

    # forward
    for _ in range(iters):
        compyute_optim.step()
        torch_optim.step()

    assert is_equal(compyute_x, torch_x, tol=1e-4)
    assert is_equal(compyute_x.grad, torch_x.grad)


@pytest.mark.parametrize("shape", shape_testdata)
@pytest.mark.parametrize("lr", lr_testdata)
@pytest.mark.parametrize("beta1", beta1_testdata)
@pytest.mark.parametrize("beta2", beta2_testdata)
@pytest.mark.parametrize("eps", eps_testdata)
@pytest.mark.parametrize("weight_decay", weight_decay_testdata)
def test_adam(shape, lr, beta1, beta2, eps, weight_decay) -> None:
    """Test for the adam optimizer."""

    # init parameters
    compyute_x, torch_x = get_random_params(shape)
    compyute_dx, torch_dx = get_random_floats(shape, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    # init compyute optimizer
    compyute_optim = Adam([compyute_x], lr, beta1, beta2, eps, weight_decay)

    # init torch optimizer
    torch_optim = torch.optim.Adam([torch_x], lr, (beta1, beta2), eps, weight_decay)

    # forward
    for _ in range(iters):
        compyute_optim.step()
        torch_optim.step()

    assert is_equal(compyute_x, torch_x)
    assert is_equal(compyute_x.grad, torch_x.grad)


@pytest.mark.parametrize("shape", shape_testdata)
@pytest.mark.parametrize("lr", lr_testdata)
@pytest.mark.parametrize("beta1", beta1_testdata)
@pytest.mark.parametrize("beta2", beta2_testdata)
@pytest.mark.parametrize("eps", eps_testdata)
@pytest.mark.parametrize("weight_decay", weight_decay_testdata)
def test_adamw(shape, lr, beta1, beta2, eps, weight_decay) -> None:
    """Test for the adamW optimizer."""

    # init parameters
    compyute_x, torch_x = get_random_params(shape)
    compyute_dx, torch_dx = get_random_floats(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    # init compyute optimizer
    compyute_optim = AdamW([compyute_x], lr, beta1, beta2, eps, weight_decay)

    # init torch optimizer
    torch_optim = torch.optim.AdamW([torch_x], lr, (beta1, beta2), eps, weight_decay)

    # forward
    for _ in range(iters):
        compyute_optim.step()
        torch_optim.step()

    assert is_equal(compyute_x, torch_x)
    assert is_equal(compyute_x.grad, torch_x.grad)


@pytest.mark.parametrize("shape", shape_testdata)
@pytest.mark.parametrize("lr", lr_testdata)
@pytest.mark.parametrize("beta1", beta1_testdata)
@pytest.mark.parametrize("beta2", beta2_testdata)
@pytest.mark.parametrize("eps", eps_testdata)
@pytest.mark.parametrize("weight_decay", weight_decay_testdata)
@pytest.mark.parametrize("momentum_decay", [2e-3, 4e-3])
def test_nadam(shape, lr, beta1, beta2, eps, weight_decay, momentum_decay) -> None:
    """Test for the nadam optimizer."""

    # init parameters
    compyute_x, torch_x = get_random_params(shape)
    compyute_dx, torch_dx = get_random_floats(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx
    torch_x.grad = torch_dx

    # init compyute optimizer
    compyute_optim = NAdam(
        [compyute_x], lr, beta1, beta2, eps, weight_decay, momentum_decay
    )

    # init torch optimizer
    torch_optim = torch.optim.NAdam(
        [torch_x], lr, (beta1, beta2), eps, weight_decay, momentum_decay
    )

    # forward
    for _ in range(iters):
        compyute_optim.step()
        torch_optim.step()

    assert is_equal(compyute_x, torch_x)
    assert is_equal(compyute_x.grad, torch_x.grad)
