"""Compyute types module"""

from typing import Literal, TypeAlias

import cupy
import numpy

_ArrayLike = numpy.ndarray | cupy.ndarray

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
_ShapeLike: TypeAlias = tuple[int, ...]
_AxisLike: TypeAlias = int | tuple[int, ...]
_DeviceLike = Literal["cpu", "cuda"]
_DtypeLike = Literal[
    "int8", "int16", "int32", "int64", "float16", "float32", "float64", "complex64", "complex128"
]


class ShapeError(Exception):
    """Incompatible tensor shapes."""
