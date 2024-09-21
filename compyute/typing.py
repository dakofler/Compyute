"""Tensor data types."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, TypeAlias

# import ml_dtypes
import numpy

# bfloat16 does not work with a lot of things yet

__all__ = [
    "bool_",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    # "bfloat16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "use_dtype",
]


@dataclass(repr=False)
class DType:

    t: type

    def __repr__(self) -> str:
        return f"DType({self.t.__name__})"


bool_ = DType(numpy.bool_)
bool_.__doc__ = "Boolean."

int8 = DType(numpy.int8)
int8.__doc__ = "Signed 8 bit integer."

int16 = DType(numpy.int16)
int16.__doc__ = "Signed 16 bit integer."

int32 = DType(numpy.int32)
int32.__doc__ = "Signed 32 bit integer."

int64 = DType(numpy.int64)
int64.__doc__ = "Signed 64 bit integer."

float16 = DType(numpy.float16)
float16.__doc__ = "16 bit floating point."

# bfloat16 = DType(ml_dtypes.bfloat16)
# bfloat16.__doc__ = "16 bit brain floating point."

float32 = DType(numpy.float32)
float32.__doc__ = "32 bit floating point."

float64 = DType(numpy.float64)
float64.__doc__ = "64 bit floating point."

complex64 = DType(numpy.complex64)
complex64.__doc__ = "Complex 64 bit floating point."

complex128 = DType(numpy.complex128)
complex128.__doc__ = "Complex 128 bit floating point."


DTYPES = {
    "bool": bool_,
    "int8": int8,
    "int16": int16,
    "int32": int32,
    "int64": int64,
    "float16": float16,
    # "bfloat16": bfloat16,
    "float32": float32,
    "float64": float64,
    "complex64": complex64,
    "complex128": complex128,
}

FLOAT_DTYPES = tuple(d for d in DTYPES.values() if "float" in d.t.__name__)
INT_DTYPES = tuple(d for d in DTYPES.values() if "int" in d.t.__name__)
COMPLEX_DTYPES = tuple(d for d in DTYPES.values() if "complex" in d.t.__name__)

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
