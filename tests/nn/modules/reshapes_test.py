"""Reshape module tests"""

import math

import pytest

from compyute.nn import Flatten
from tests.utils import get_random_floats


@pytest.mark.parametrize("shape", [(8, 16), (8, 16, 32), (8, 16, 32, 64)])
def test_flatten(shape) -> None:
    """Test for the flatten layer."""

    # init compyute module
    compyute_module = Flatten()

    # forward
    compyute_x, _ = get_random_floats(shape)
    with compyute_module.train():
        compyute_y = compyute_module(compyute_x)
    assert compyute_y.shape == (compyute_x.shape[0], math.prod(shape[1:]))

    # backward
    compyute_dy, _ = get_random_floats(compyute_y.shape)
    with compyute_module.train():
        compyute_dx = compyute_module.backward(compyute_dy)

    assert compyute_dx.shape == shape
