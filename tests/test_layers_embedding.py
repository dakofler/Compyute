"""Embedding layer tests"""

import torch

from src.compyute.nn import Embedding
from src.compyute.random import uniform_int
from tests.test_utils import get_params, get_vals_float, validate

B, Cin, Cout, X = (10, 20, 30, 40)


def test_embedding() -> None:
    """Test for the embedding layer."""
    results = []
    shape_w = (Cin, Cout)

    # forward
    compyute_x = uniform_int((B, X), 0, Cin)
    torch_x = torch.from_numpy(compyute_x.to_numpy())
    compyute_w, torch_w = get_params(shape_w)

    compyute_module = Embedding(Cin, Cout)
    compyute_module.set_training(True)
    compyute_module.w = compyute_w
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.Embedding(Cin, Cout)
    torch_module.weight = torch_w
    torch_y = torch_module(torch_x)

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals_float(compyute_y.shape, torch_grad=False)
    _ = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)

    results.append(validate(compyute_module.w.grad, torch_module.weight.grad))

    assert all(results)
