"""Regularization module tests"""

import compyute
from compyute.nn import Dropout
from tests.test_utils import get_random_floats

SHAPE = (10, 20, 30)


def test_dropout() -> None:
    """Test for the dropout layer."""
    shape_x = SHAPE

    # init compyute module
    compyute_module = Dropout(p=0.5)

    # forward
    compyute_x, _ = get_random_floats(shape_x)
    with compyute_module.train():
        compyute_y = compyute_module(compyute_x)

    # backward
    compyute_dy, _ = get_random_floats(compyute_y.shape)
    with compyute_module.train():
        compyute_dx = compyute_module.backward(compyute_dy)

    assert compyute_x.to_type(compyute.bool) == compyute_dx.to_type(compyute.bool)
    assert compyute_dx == 2 * compyute_dy
