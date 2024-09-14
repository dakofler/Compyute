"""Tensor typing tests"""

import compyute as cp

# data types should be prioritized as follows:
# tensor(): specified > default > inferred from data
# others: specified > default > fallback


def test_tensor() -> None:
    """Test for the tensor."""

    x = cp.tensor([1, 2, 3])
    assert x.dtype == cp.int64

    x = cp.tensor([1.0, 2.0, 3.0])
    assert x.dtype == cp.float64

    x = cp.tensor([1, 2, 3], dtype=cp.float32)
    assert x.dtype == cp.float32

    x = cp.tensor([1.0, 2.0, 3.0], dtype=cp.int32)
    assert x.dtype == cp.int32

    with cp.use_dtype(cp.float64):
        x = cp.tensor([1, 2, 3])
    assert x.dtype == cp.float64

    with cp.use_dtype(cp.float64):
        x = cp.tensor([1, 2, 3], dtype=cp.int32)
    assert x.dtype == cp.int32


def test_creation() -> None:
    """Test for the creation functions."""

    x = cp.random.normal((10, 10))
    assert x.dtype == cp.float32

    with cp.use_dtype(cp.int8):
        x = cp.random.normal((10, 10))
    assert x.dtype == cp.int8

    with cp.use_dtype(cp.int8):
        x = cp.random.normal((10, 10), dtype=cp.int32)
    assert x.dtype == cp.int32
