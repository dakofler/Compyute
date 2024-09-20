"""Regularization module tests"""

import pytest

import compyute
from compyute.nn import Dropout
from tests.utils import get_random_floats


@pytest.mark.parametrize("shape", [(8, 16), (8, 16, 32), (8, 16, 32, 64)])
@pytest.mark.parametrize("p", [0.1, 0.2, 0.5])
def test_dropout(shape, p) -> None:
    """Test for the dropout layer."""

    # init compyute module
    compyute_module = Dropout(p=p)

    # forward
    compyute_x, _ = get_random_floats(shape)
    compyute_y = compyute_module(compyute_x)

    # backward
    compyute_dy, _ = get_random_floats(compyute_y.shape)
    compyute_dx = compyute_module.backward(compyute_dy)

    assert compyute_x.to_type(compyute.bool_) == compyute_dx.to_type(compyute.bool_)
    assert compyute_dx == 1 / p * compyute_dy
