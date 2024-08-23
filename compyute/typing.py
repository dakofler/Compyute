"""Tensor data types."""

from contextlib import contextmanager
from typing import Optional, TypeAlias

import numpy

__all__ = [
    "bool_",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "use_dtype",
]


bool_: TypeAlias = numpy.bool_
int8: TypeAlias = numpy.int8
int16: TypeAlias = numpy.int16
int32: TypeAlias = numpy.int32
int64: TypeAlias = numpy.int64
float16: TypeAlias = numpy.float16
float32: TypeAlias = numpy.float32
float64: TypeAlias = numpy.float64
complex64: TypeAlias = numpy.complex64
complex128: TypeAlias = numpy.complex128


DTYPES = {
    bool_,
    int8,
    int16,
    int32,
    int64,
    float16,
    float32,
    float64,
    complex64,
    complex128,
}
FLOAT_DTYPES = {d for d in DTYPES if "float" in str(d)}
INT_DTYPES = {d for d in DTYPES if "int" in str(d)}
COMPLEX_DTYPES = {d for d in DTYPES if "complex" in str(d)}

ScalarLike: TypeAlias = int | float | complex
from numpy.typing import DTypeLike


def is_integer(dtype: DTypeLike) -> bool:
    """Returns ``True`` if the data type is an integer."""
    return numpy.issubdtype(dtype, numpy.integer)


def is_float(dtype: DTypeLike) -> bool:
    """Returns ``True`` if the data type is a float."""
    return numpy.issubdtype(dtype, numpy.floating)


default_dtype = float32


def set_default_dtype(dtype: DTypeLike) -> None:
    """Sets the default data type."""
    global default_dtype
    default_dtype = dtype


@contextmanager
def use_dtype(dtype: DTypeLike):
    """Context manager to set the default dtype when creating tensors."""
    set_default_dtype(dtype)
    try:
        yield
    finally:
        set_default_dtype(dtype)


def select_dtype(dtype: Optional[DTypeLike]) -> DTypeLike:
    """Selects the data type. Returns the default data type if dtype is ``None``."""
    return dtype or default_dtype
