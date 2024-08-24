"""Tensor data types."""

from contextlib import contextmanager
from enum import Enum
from typing import Optional, TypeAlias

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


class DType(Enum):
    """Data type enum."""

    BOOL = "bool"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"

    def __repr__(self) -> str:
        return f"compyute.{self.value}"

    def __str__(self) -> str:
        return self.__repr__()


bool_ = DType.BOOL
bool_.__doc__ = "Boolean."
int8 = DType.INT8
int8.__doc__ = "Signed 8 bit integer."
int16 = DType.INT16
int16.__doc__ = "Signed 16 bit integer."
int32 = DType.INT32
int32.__doc__ = "Signed 32 bit integer."
int64 = DType.INT64
int64.__doc__ = "Signed 64 bit integer."
float16 = DType.FLOAT16
float16.__doc__ = "16 bit floating point."
float32 = DType.FLOAT32
float32.__doc__ = "32 bit floating point."
float64 = DType.FLOAT64
float64.__doc__ = "64 bit floating point."
complex64 = DType.COMPLEX64
complex64.__doc__ = "Complex 64 bit floating point."
complex128 = DType.COMPLEX128
complex128.__doc__ = "Complex 128 bit floating point."


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


def is_integer(dtype: DType) -> bool:
    """Returns ``True`` if the data type is an integer."""
    return dtype in INT_DTYPES


def is_float(dtype: DType) -> bool:
    """Returns ``True`` if the data type is a float."""
    return dtype in FLOAT_DTYPES


default_dtype = float32


def set_default_dtype(dtype: DType) -> None:
    """Sets the default data type."""
    global default_dtype
    default_dtype = dtype


@contextmanager
def use_dtype(dtype: DType):
    """Context manager to set the default dtype when creating tensors."""
    set_default_dtype(dtype)
    try:
        yield
    finally:
        set_default_dtype(dtype)


def select_dtype(dtype: Optional[DType]) -> DType:
    """Selects the data type. Returns the default data type if dtype is ``None``."""
    return dtype or default_dtype
