"""Tensor data types."""

from contextlib import contextmanager
from enum import StrEnum, auto
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


class DType(StrEnum):
    """Data type enum."""

    BOOL = auto()
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
    INT64 = auto()
    FLOAT16 = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()
    COMPLEX64 = auto()
    COMPLEX128 = auto()

    def __repr__(self) -> str:
        return "compyute." + self.value

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


fallback_default_dtype: DType = float32
default_dtype: Optional[DType] = None


def set_default_dtype(dtype: Optional[DType]) -> None:
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
        set_default_dtype(None)


def get_default_dtype() -> Optional[DType]:
    """Returns the default data type."""
    return default_dtype


def select_dtype(dtype: Optional[DType]) -> DType:
    """Selects the data type. Returns the default data type if dtype is ``None``."""
    return dtype or default_dtype or fallback_default_dtype
