"""Embedding layer tests"""

import torch
import walnut
from tests.test_utils import get_vals, get_params, validate


B, Cin, Cout, X = (10, 10, 10, 10)


# Embedding
def test_embedding_cpu() -> None:
    results = []
    shape_w = (Cin, Cout)

    # forward
    walnut_x = walnut.randint((B, X), 0, Cin)
    torch_x = torch.from_numpy(walnut_x.data)
    walnut_w, torch_w = get_params(shape_w)

    walnut_module = walnut.nn.layers.Embedding(Cin, Cout)
    walnut_module.training = True
    walnut_module.w = walnut_w
    walnut_y = walnut_module(walnut_x)

    torch_module = torch.nn.Embedding(Cin, Cout)
    torch_module.weight = torch_w
    torch_y = torch_module(torch_x)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False)
    _ = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(walnut_module.w.grad, torch_module.weight.grad))

    assert all(results)


def test_embedding_cuda() -> None:
    results = []
    shape_w = (Cin, Cout)

    # forward
    walnut_x = walnut.randint((B, X), 0, Cin)
    torch_x = torch.from_numpy(walnut_x.data)
    walnut_x.to_device("cuda")
    walnut_w, torch_w = get_params(shape_w, device="cuda")

    walnut_module = walnut.nn.layers.Embedding(Cin, Cout)
    walnut_module.training = True
    walnut_module.to_device("cuda")
    walnut_module.w = walnut_w
    walnut_y = walnut_module(walnut_x)

    torch_module = torch.nn.Embedding(Cin, Cout)
    torch_module.weight = torch_w
    torch_y = torch_module(torch_x)

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False, device="cuda")
    _ = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(walnut_module.w.grad, torch_module.weight.grad))

    assert all(results)
