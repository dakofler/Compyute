"""Compyute engine utilities."""

import os
from enum import Enum
from functools import cache
from types import ModuleType
from typing import Literal, TypeAlias

import cupy
import numpy

__all__ = ["cpu", "cuda", "gpu_available", "set_cuda_tf32"]


class Device(Enum):
    """Device enum."""

    CPU = "cpu"
    CUDA = "cuda"

    def __repr__(self) -> str:
        return f"Dtype('{self.value}')"

    def __str__(self) -> str:
        return self.__repr__()


cpu = Device.CPU
cuda = Device.CUDA


@cache
def _cuda_available() -> bool:
    try:
        return cupy.is_available()
    except Exception:
        return False


_ArrayLike: TypeAlias = numpy.ndarray | cupy.ndarray
_DeviceLike: TypeAlias = Literal["cpu", "cuda"] | Device

_AVAILABLE_DEVICES = {Device.CPU}.union({Device.CUDA} if _cuda_available() else {})
_DEVICE_TO_ENGINE: dict[Device, ModuleType] = {Device.CPU: numpy, Device.CUDA: cupy}
_ENGINE_TO_DEVICE: dict[type, Device] = {numpy.ndarray: Device.CPU, cupy.ndarray: Device.CUDA}


def gpu_available() -> bool:
    """Checks, whether one or more GPUs are available.

    Returns
    -------
    bool
        True, if one or more GPUs are available.
    """
    return Device.CUDA in _AVAILABLE_DEVICES


@cache
def available(device: Device) -> None:
    """Checks if the specified device is available."""
    if device not in _AVAILABLE_DEVICES:
        raise AttributeError(f"Device {device} is not available.")


@cache
def get_engine(device: _DeviceLike) -> ModuleType:
    """Returns the computation engine for a given device."""
    device = Device(device)
    available(device)
    return _DEVICE_TO_ENGINE.get(device, numpy)


@cache
def infer_device(array_type: type) -> Device:
    """Infers the device by type."""
    return _ENGINE_TO_DEVICE.get(array_type, Device.CPU)


def data_to_device(data: _ArrayLike, device: Device) -> _ArrayLike:
    """Moves the data to the specified device."""
    if device == Device.CPU:
        return cupy.asnumpy(data)
    available(device)
    return cupy.array(data)


def get_array_string(array: _ArrayLike) -> str:
    """Returns the array as a formatted string."""
    return numpy.array2string(
        array,
        max_line_width=100,
        prefix="Tensor(",
        separator=", ",
        precision=4,
        floatmode="maxprec_equal",
    )


def set_cuda_tf32(value: bool) -> None:
    """Allows CUDA libraries to use Tensor Cores TF32 compute for 32-bit floating point compute.

    Parameters
    ----------
    value : bool
        Whether TF32 compute should be enabled.
    """
    if value:
        os.environ["CUPY_TF32"] = "1"
    else:
        del os.environ["CUPY_TF32"]
