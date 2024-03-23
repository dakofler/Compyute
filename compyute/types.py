"""Compyute types module"""

from typing import Literal
import numpy
import cupy

ArrayLike = numpy.ndarray | cupy.ndarray

IntLike = (
    numpy.int8
    | numpy.int16
    | numpy.int32
    | numpy.int64
    | cupy.int8
    | cupy.int16
    | cupy.int32
    | cupy.int64
    | cupy.int64
    | Literal["int", "int8", "int16", "int32", "int64"]
)

FloatLike = (
    numpy.float16
    | numpy.float32
    | numpy.float64
    | cupy.float16
    | cupy.float16
    | cupy.float32
    | cupy.float64
    | Literal["float", "float16", "float32", "float64"]
)

ComplexLike = (
    numpy.complex64
    | numpy.complex128
    | cupy.complex64
    | cupy.complex128
    | Literal["complex", "complex64", "complex128"]
)

DtypeLike = IntLike | FloatLike | ComplexLike

ScalarLike = DtypeLike | list | float | int
ShapeLike = tuple[int, ...]
AxisLike = int | tuple[int, ...]
DeviceLike = Literal["cpu", "cuda"]
