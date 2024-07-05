"""Embedding layer tests"""

import torch

from src.compyute.nn import Embedding
from tests.test_utils import get_random_floats, get_random_integers, get_random_params, is_equal

B, Cin, Cout, X = (10, 20, 30, 40)


def test_embedding() -> None:
    """Test for the embedding layer."""
    shape_x = (B, X)
    shape_w = (Cin, Cout)

    # init parameters
    compyute_w, torch_w = get_random_params(shape_w)

    # init compyute module
    compyute_module = Embedding(Cin, Cout, training=True)
    compyute_module.w = compyute_w

    # init torch module
    torch_module = torch.nn.Embedding(Cin, Cout)
    torch_module.weight = torch_w

    # forward
    compyute_x, torch_x = get_random_integers(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_equal(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    _ = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_equal(compyute_module.w.grad, torch_module.weight.grad)
