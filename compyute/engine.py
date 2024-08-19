"""Compyute engine utilities."""

from enum import Enum
from functools import cache
from types import ModuleType
from typing import Literal, Optional, TypeAlias

import cupy as CUDA_ENGINE
import numpy as CPU_ENGINE

__all__ = ["cpu", "cuda", "gpu_available"]


class Device(Enum):
    """Device enum."""

    CPU = "cpu"
    CUDA = "cuda"

    def __repr__(self) -> str:
        return f"compyute.{self.value}"

    def __str__(self) -> str:
        return self.__repr__()


cpu = Device.CPU
cuda = Device.CUDA


@cache
def _cuda_available() -> bool:
    try:
        return CUDA_ENGINE.is_available()
    except Exception:
        return False


_ArrayLike: TypeAlias = CPU_ENGINE.ndarray | CUDA_ENGINE.ndarray
_DeviceLike: TypeAlias = Literal["cpu", "cuda"] | Device


AVAILABLE_DEVICES = {Device.CPU, Device.CUDA} if _cuda_available() else {Device.CPU}


def gpu_available() -> bool:
    """Checks, whether one or more GPUs are available.

    Returns
    -------
    bool
        True, if one or more GPUs are available.
    """
    return Device.CUDA in AVAILABLE_DEVICES


@cache
def available(device: Device) -> None:
    """Checks if the specified device is available."""
    if device not in AVAILABLE_DEVICES:
        raise AttributeError(f"Device {device} is not available.")


@cache
def get_engine(device: Optional[_DeviceLike]) -> ModuleType:
    """Returns the computation engine for a given device."""
    if device is None:
        return CPU_ENGINE

    device = Device(device)
    if device == Device.CPU:
        return CPU_ENGINE

    available(device)
    return CUDA_ENGINE


@cache
def get_device(array_type: type) -> Device:
    """Infers the device by type."""
    if array_type == CUDA_ENGINE.ndarray:
        return Device.CUDA
    return Device.CPU


def data_to_device(data: _ArrayLike, device: Device) -> _ArrayLike:
    """Moves the data to the specified device."""
    if device == Device.CPU:
        return CUDA_ENGINE.asnumpy(data)
    available(device)
    return CUDA_ENGINE.array(data)


def get_array_string(array: _ArrayLike) -> str:
    """Returns the array as a formatted string."""
    return CPU_ENGINE.array2string(
        array,
        max_line_width=100,
        prefix="Tensor(",
        separator=", ",
        precision=4,
        floatmode="maxprec_equal",
    )
