"""Compyute data type utilities."""

import os
from contextlib import contextmanager
from enum import Enum
from typing import Literal, Optional, TypeAlias

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
    "get_default_dtype",
    "set_default_dtype",
    "default_dtype",
]


class Dtype(Enum):
    """Compyute data type."""

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
_DtypeLike: TypeAlias = (
    Dtype
    | Literal[
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
)


def dtype_to_str(
    dtype: _DtypeLike,
) -> Literal[
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
]:
    """Returns the string representation of a dtype."""
    return Dtype(dtype).value


def set_default_dtype(dtype: Optional[_DtypeLike] = None) -> None:
    """Sets the default data type for new tensors.

    Parameters
    ----------
    device : _DeviceLike, optional
        The data type new tensors will be using. Defaults to ``None``.
        If ``None``, the default data type will be deleted.
    """
    if dtype is None:
        del os.environ["COMPYUTE_DEFAULT_DTYPE"]
        return
    os.environ["COMPYUTE_DEFAULT_DTYPE"] = Dtype(dtype).value


@contextmanager
def default_dtype(dtype: _DtypeLike):
    """Context manager to set the default data type for new tensors.

    Parameters
    ----------
    dtype : _DtypeLike
        The data type new tensors will be using.
    """
    set_default_dtype(dtype)
    try:
        yield
    finally:
        set_default_dtype()


def get_default_dtype() -> Optional[Dtype]:
    """Returns the default data type for new tensors.

    Returns
    -------
    Dtype | None
        The default data type for new tensors. ``None`` if no default data type is set.
    """
    if "COMPYUTE_DEFAULT_DTYPE" in os.environ:
        return Dtype(os.environ["COMPYUTE_DEFAULT_DTYPE"])
    return None


def select_dtype(dtype: Optional[_DtypeLike]) -> Optional[Dtype]:
    """Chooses a data type based on available options.
    - If ``dtype`` is not ``None``, it is returned.
    - If ``dtype`` is ``None`` and a default data type is set, the default data type is returned.
    - If no default data type is set, ``None`` is returned.

    Parameters
    ----------
    dtype : _DtypeLike | None
        The data type to select.

    Returns
    -------
    Dtype | None
        The selected data type.
    """
    if dtype is not None:
        return Dtype(dtype)
    return get_default_dtype()


def select_dtype_or_float(dtype: Optional[_DtypeLike]) -> Dtype:
    """Chooses a data type based on available options.
    - If ``dtype`` is not ``None``, it is returned.
    - If ``dtype`` is ``None`` and a default data type is set, the default data type is returned.
    - If no default data type is set, ``None`` is returned.

    Parameters
    ----------
    dtype : _DtypeLike | None
        The data type to select.

    Returns
    -------
    Dtype | None
        The selected data type.
    """
    if dtype is not None:
        return Dtype(dtype)
    return get_default_dtype() or Dtype.FLOAT32


def select_dtype_str(dtype: Optional[_DtypeLike]) -> Optional[str]:
    """Chooses a data type based on available options and returns a string."""
    dtype = select_dtype(dtype)
    return None if dtype is None else dtype.value
