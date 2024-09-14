"""Embedding module tests"""

import pytest
import torch

from compyute.nn import Embedding
from tests.utils import (
    get_random_floats,
    get_random_integers,
    get_random_params,
    is_close,
)

testdata = [(8, 16, 32), (8, 16, 32, 64), (8, 16, 32, 64, 128)]


@pytest.mark.parametrize("shape", testdata)
def test_embedding(shape) -> None:
    """Test for the embedding layer."""

    shape_x = shape[:-2]  # (B1, ..., Bn)
    shape_w = shape[-2:]  # (Cin, Cout)

    # init parameters
    compyute_w, torch_w = get_random_params(shape_w)

    # init compyute module
    compyute_module = Embedding(*shape_w)
    compyute_module.w = compyute_w

    # init torch module
    torch_module = torch.nn.Embedding(*shape_w)
    torch_module.weight = torch_w

    # forward
    compyute_x, torch_x = get_random_integers(shape_x)
    with compyute_module.train():
        compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    with compyute_module.train():
        _ = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_module.w.grad, torch_module.weight.grad)
