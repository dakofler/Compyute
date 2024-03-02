"""Optimizers tests"""

import torch
import compyute
from tests.test_utils import get_vals, get_params, validate


SHAPE = (10, 10)
ITER = 10


# SGD
def test_sgd_cpu() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals(compyute_x.shape, torch_grad=False)
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


def test_sgd_cuda() -> None:
    if not compyute.cuda.is_available():
        pass
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE, device="cuda")
    compyute_dx, torch_dx = get_vals(compyute_x.shape, torch_grad=False, device="cuda")
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


def test_sgd_m_cpu() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals(compyute_x.shape, torch_grad=False)
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


def test_sgd_m_nesterov_cpu() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals(compyute_x.shape, torch_grad=False)
    compyute_x.grad = compyute_dx.data
    torch_x.grad = torch_dx

    compyute_optim = compyute.nn.optimizers.SGD(lr=1e-2, m=0.1, nesterov=True)
    compyute_optim.parameters = [compyute_x]

    torch_optim = torch.optim.SGD([torch_x], lr=1e-2, momentum=0.1, nesterov=True)

    for _ in range(ITER):
        compyute_optim.step()
        torch_optim.step()

    results.append(validate(compyute_x, torch_x))
    results.append(validate(compyute_x.grad, torch_x.grad))

    assert all(results)


def test_sgd_m_wdecay_cpu() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals(compyute_x.shape, torch_grad=False)
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


# Adam
def test_adam_cpu() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals(compyute_x.shape, torch_grad=False)
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


def test_adam_cuda() -> None:
    if not compyute.cuda.is_available():
        pass
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE, device="cuda")
    compyute_dx, torch_dx = get_vals(compyute_x.shape, torch_grad=False, device="cuda")
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


def test_adam_wdecay_cpu() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals(compyute_x.shape, torch_grad=False)
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


# AdamW
def test_adamw_cpu() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals(compyute_x.shape, torch_grad=False)
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


def test_adamw_cuda() -> None:
    if not compyute.cuda.is_available():
        pass
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE, device="cuda")
    compyute_dx, torch_dx = get_vals(compyute_x.shape, torch_grad=False, device="cuda")
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


def test_adamw_wdecay_cpu() -> None:
    results = []

    # forward
    compyute_x, torch_x = get_params(SHAPE)
    compyute_dx, torch_dx = get_vals(compyute_x.shape, torch_grad=False)
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
