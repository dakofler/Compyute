"""Optimizers tests"""

import torch
import walnut
from tests.test_utils import get_vals, get_params, validate


SHAPE = (10, 10)
ITER = 10


# SGD
def test_sgd_cpu() -> None:
    results = []

    # forward
    walnut_x, torch_x = get_params(SHAPE)
    walnut_dx, torch_dx = get_vals(walnut_x.shape, torch_grad=False)
    walnut_x.grad = walnut_dx.data
    torch_x.grad = torch_dx

    walnut_optim = walnut.nn.optimizers.SGD(l_r=1e-2)
    walnut_optim.parameters = [walnut_x]

    torch_optim = torch.optim.SGD([torch_x], lr=1e-2)

    for _ in range(ITER):
        walnut_optim.step()
        torch_optim.step()

    results.append(validate(walnut_x, torch_x))
    results.append(validate(walnut_x.grad, torch_x.grad))

    assert all(results)


def test_sgd_cuda() -> None:
    if not walnut.cuda.is_available():
        pass
    results = []

    # forward
    walnut_x, torch_x = get_params(SHAPE, device="cuda")
    walnut_dx, torch_dx = get_vals(walnut_x.shape, torch_grad=False, device="cuda")
    walnut_x.grad = walnut_dx.data
    torch_x.grad = torch_dx

    walnut_optim = walnut.nn.optimizers.SGD(l_r=1e-2)
    walnut_optim.parameters = [walnut_x]

    torch_optim = torch.optim.SGD([torch_x], lr=1e-2)

    for _ in range(ITER):
        walnut_optim.step()
        torch_optim.step()

    results.append(validate(walnut_x, torch_x))
    results.append(validate(walnut_x.grad, torch_x.grad))

    assert all(results)


def test_sgd_m_cpu() -> None:
    results = []

    # forward
    walnut_x, torch_x = get_params(SHAPE)
    walnut_dx, torch_dx = get_vals(walnut_x.shape, torch_grad=False)
    walnut_x.grad = walnut_dx.data
    torch_x.grad = torch_dx

    walnut_optim = walnut.nn.optimizers.SGD(l_r=1e-2, m=0.1)
    walnut_optim.parameters = [walnut_x]

    torch_optim = torch.optim.SGD([torch_x], lr=1e-2, momentum=0.1)

    for _ in range(ITER):
        walnut_optim.step()
        torch_optim.step()

    results.append(validate(walnut_x, torch_x))
    results.append(validate(walnut_x.grad, torch_x.grad))

    assert all(results)


def test_sgd_m_nesterov_cpu() -> None:
    results = []

    # forward
    walnut_x, torch_x = get_params(SHAPE)
    walnut_dx, torch_dx = get_vals(walnut_x.shape, torch_grad=False)
    walnut_x.grad = walnut_dx.data
    torch_x.grad = torch_dx

    walnut_optim = walnut.nn.optimizers.SGD(l_r=1e-2, m=0.1, nesterov=True)
    walnut_optim.parameters = [walnut_x]

    torch_optim = torch.optim.SGD([torch_x], lr=1e-2, momentum=0.1, nesterov=True)

    for _ in range(ITER):
        walnut_optim.step()
        torch_optim.step()

    results.append(validate(walnut_x, torch_x))
    results.append(validate(walnut_x.grad, torch_x.grad))

    assert all(results)


def test_sgd_m_wdecay_cpu() -> None:
    results = []

    # forward
    walnut_x, torch_x = get_params(SHAPE)
    walnut_dx, torch_dx = get_vals(walnut_x.shape, torch_grad=False)
    walnut_x.grad = walnut_dx.data
    torch_x.grad = torch_dx

    walnut_optim = walnut.nn.optimizers.SGD(l_r=1e-2, m=0.1, weight_decay=0.1)
    walnut_optim.parameters = [walnut_x]
    torch_optim = torch.optim.SGD([torch_x], lr=1e-2, momentum=0.1, weight_decay=0.1)

    for _ in range(ITER):
        walnut_optim.step()
        torch_optim.step()

    results.append(validate(walnut_x, torch_x))
    results.append(validate(walnut_x.grad, torch_x.grad))

    assert all(results)


# Adam
def test_adam_cpu() -> None:
    results = []

    # forward
    walnut_x, torch_x = get_params(SHAPE)
    walnut_dx, torch_dx = get_vals(walnut_x.shape, torch_grad=False)
    walnut_x.grad = walnut_dx.data
    torch_x.grad = torch_dx

    walnut_optim = walnut.nn.optimizers.Adam(l_r=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)
    walnut_optim.parameters = [walnut_x]

    torch_optim = torch.optim.Adam([torch_x], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    for _ in range(ITER):
        walnut_optim.step()
        torch_optim.step()

    results.append(validate(walnut_x, torch_x))
    results.append(validate(walnut_x.grad, torch_x.grad))

    assert all(results)


def test_adam_cuda() -> None:
    if not walnut.cuda.is_available():
        pass
    results = []

    # forward
    walnut_x, torch_x = get_params(SHAPE, device="cuda")
    walnut_dx, torch_dx = get_vals(walnut_x.shape, torch_grad=False, device="cuda")
    walnut_x.grad = walnut_dx.data
    torch_x.grad = torch_dx

    walnut_optim = walnut.nn.optimizers.Adam(l_r=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)
    walnut_optim.parameters = [walnut_x]

    torch_optim = torch.optim.Adam([torch_x], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    for _ in range(ITER):
        walnut_optim.step()
        torch_optim.step()

    results.append(validate(walnut_x, torch_x))
    results.append(validate(walnut_x.grad, torch_x.grad))

    assert all(results)


def test_adam_wdecay_cpu() -> None:
    results = []

    # forward
    walnut_x, torch_x = get_params(SHAPE)
    walnut_dx, torch_dx = get_vals(walnut_x.shape, torch_grad=False)
    walnut_x.grad = walnut_dx.data
    torch_x.grad = torch_dx

    walnut_optim = walnut.nn.optimizers.Adam(
        l_r=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1
    )
    walnut_optim.parameters = [walnut_x]

    torch_optim = torch.optim.Adam(
        [torch_x], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1
    )

    for _ in range(ITER):
        walnut_optim.step()
        torch_optim.step()

    results.append(validate(walnut_x, torch_x))
    results.append(validate(walnut_x.grad, torch_x.grad))
    assert all(results)


# AdamW
def test_adamw_cpu() -> None:
    results = []

    # forward
    walnut_x, torch_x = get_params(SHAPE)
    walnut_dx, torch_dx = get_vals(walnut_x.shape, torch_grad=False)
    walnut_x.grad = walnut_dx.data
    torch_x.grad = torch_dx

    walnut_optim = walnut.nn.optimizers.AdamW(
        l_r=1e-3, beta1=0.9, beta2=0.999, eps=1e-8
    )
    walnut_optim.parameters = [walnut_x]

    torch_optim = torch.optim.AdamW([torch_x], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    for _ in range(ITER):
        walnut_optim.step()
        torch_optim.step()

    results.append(validate(walnut_x, torch_x))
    results.append(validate(walnut_x.grad, torch_x.grad))

    assert all(results)


def test_adamw_cuda() -> None:
    if not walnut.cuda.is_available():
        pass
    results = []

    # forward
    walnut_x, torch_x = get_params(SHAPE, device="cuda")
    walnut_dx, torch_dx = get_vals(walnut_x.shape, torch_grad=False, device="cuda")
    walnut_x.grad = walnut_dx.data
    torch_x.grad = torch_dx

    walnut_optim = walnut.nn.optimizers.AdamW(
        l_r=1e-3, beta1=0.9, beta2=0.999, eps=1e-8
    )
    walnut_optim.parameters = [walnut_x]

    torch_optim = torch.optim.AdamW([torch_x], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    for _ in range(ITER):
        walnut_optim.step()
        torch_optim.step()

    results.append(validate(walnut_x, torch_x))
    results.append(validate(walnut_x.grad, torch_x.grad))

    assert all(results)


def test_adamw_wdecay_cpu() -> None:
    results = []

    # forward
    walnut_x, torch_x = get_params(SHAPE)
    walnut_dx, torch_dx = get_vals(walnut_x.shape, torch_grad=False)
    walnut_x.grad = walnut_dx.data
    torch_x.grad = torch_dx

    walnut_optim = walnut.nn.optimizers.AdamW(
        l_r=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1
    )
    walnut_optim.parameters = [walnut_x]

    torch_optim = torch.optim.AdamW(
        [torch_x], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1
    )

    for _ in range(ITER):
        walnut_optim.step()
        torch_optim.step()

    results.append(validate(walnut_x, torch_x))
    results.append(validate(walnut_x.grad, torch_x.grad))

    assert all(results)
