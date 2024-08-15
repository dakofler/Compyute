"""Tensor data types."""

from enum import Enum
from typing import Literal, TypeAlias

import cupy
import numpy

__all__ = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]


class Dtype(Enum):
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
        return f"Dtype('{self.value}')"

    def __str__(self) -> str:
        return self.__repr__()


DTYPES = {d.value for d in Dtype}
FLOAT_DTYPES = {d for d in DTYPES if "float" in d}
INT_DTYPES = {d for d in DTYPES if "int" in d}
COMPLEX_DTYPES = {d for d in DTYPES if "complex" in d}


bool = Dtype.BOOL
int8 = Dtype.INT8
int16 = Dtype.INT16
int32 = Dtype.INT32
int64 = Dtype.INT64
float16 = Dtype.FLOAT16
float32 = Dtype.FLOAT32
float64 = Dtype.FLOAT64
complex64 = Dtype.COMPLEX64
complex128 = Dtype.COMPLEX128

NumpyInt: TypeAlias = numpy.int8 | numpy.int16 | numpy.int32 | numpy.int64
NumpyFloat: TypeAlias = numpy.float16 | numpy.float32 | numpy.float64
NumpyComplex: TypeAlias = numpy.complex64 | numpy.complex128

CupyInt: TypeAlias = cupy.int8 | cupy.int16 | cupy.int32 | cupy.int64
CupyFloat: TypeAlias = cupy.float16 | cupy.float32 | cupy.float64
CupyComplex: TypeAlias = cupy.complex64 | cupy.complex128

_IntLike: TypeAlias = NumpyInt | CupyInt | int
_FloatLike: TypeAlias = NumpyFloat | CupyFloat | float
_ComplexLike: TypeAlias = NumpyComplex | CupyComplex | complex

_ScalarLike: TypeAlias = _IntLike | _FloatLike | _ComplexLike
_DtypeLike: TypeAlias = Dtype | Literal[*DTYPES]
