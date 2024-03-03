"""Embedding layer tests"""

import torch
import compyute
from tests.test_utils import get_vals, get_params, validate


B, Cin, Cout, X = (10, 10, 10, 10)


# Embedding
def test_embedding_cpu() -> None:
    results = []
    shape_w = (Cin, Cout)

    # forward
    compyute_x = compyute.randint((B, X), 0, Cin)
    torch_x = torch.from_numpy(compyute_x.data)
    compyute_w, torch_w = get_params(shape_w)

    compyute_module = compyute.nn.layers.Embedding(Cin, Cout)
    compyute_module.training = True
    compyute_module.w = compyute_w
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Embedding(Cin, Cout)
    torch_module.weight = torch_w
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False)
    _ = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))

    assert all(results)


def test_embedding_cuda() -> None:
    if not compyute.engine.gpu_available():
        pass
    results = []
    shape_w = (Cin, Cout)

    # forward
    compyute_x = compyute.randint((B, X), 0, Cin)
    torch_x = torch.from_numpy(compyute_x.data)
    compyute_x.to_device("cuda")
    compyute_w, torch_w = get_params(shape_w, device="cuda")

    compyute_module = compyute.nn.layers.Embedding(Cin, Cout)
    compyute_module.training = True
    compyute_module.to_device("cuda")
    compyute_module.w = compyute_w
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Embedding(Cin, Cout)
    torch_module.weight = torch_w
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False, device="cuda")
    _ = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))

    assert all(results)
