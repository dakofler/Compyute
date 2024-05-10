"""Compyute types module"""

from typing import Literal, TypeAlias
import numpy
import cupy

ArrayLike = numpy.ndarray | cupy.ndarray

NumpyInt: TypeAlias = numpy.int8 | numpy.int16 | numpy.int32 | numpy.int64
NumpyFloat: TypeAlias = numpy.float16 | numpy.float32 | numpy.float64
NumpyComplex: TypeAlias = numpy.complex64 | numpy.complex128

CupyInt: TypeAlias = cupy.int8 | cupy.int16 | cupy.int32 | cupy.int64
CupyFloat: TypeAlias = cupy.float16 | cupy.float32 | cupy.float64
CupyComplex: TypeAlias = cupy.complex64 | cupy.complex128

IntLike: TypeAlias = NumpyInt | CupyInt | int | Literal["int8", "int16", "int32", "int64"]
FloatLike: TypeAlias = NumpyFloat | CupyFloat | float | Literal["float16", "float32", "float64"]
ComplexLike: TypeAlias = NumpyComplex | CupyComplex | complex | Literal["complex64", "complex128"]

DtypeLike: TypeAlias = IntLike | FloatLike | ComplexLike
ScalarLike: TypeAlias = DtypeLike
ShapeLike: TypeAlias = tuple[int, ...]
AxisLike: TypeAlias = int | tuple[int, ...]
DeviceLike: TypeAlias = Literal["cpu", "cuda"]
