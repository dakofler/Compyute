"""Engine functions module"""

import os
from enum import Enum
from functools import cache
from types import ModuleType
from typing import Literal, TypeAlias

import cupy
import numpy

__all__ = ["cpu", "cuda"]


def _cuda_available() -> bool:
    try:
        return cupy.is_available()
    except Exception as e:
        print(e)
        return False


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


_ArrayLike: TypeAlias = numpy.ndarray | cupy.ndarray
_DeviceLike: TypeAlias = Literal["cpu", "cuda"] | Device
_AVAILABLE_DEVICES = {Device.CPU}.union({Device.CUDA} if _cuda_available() else {})
_DEVICE_TO_ENGINE: dict[Device, ModuleType] = {Device.CPU: numpy, Device.CUDA: cupy}
_ENGINE_TO_DEVICE: dict[type, Device] = {numpy.ndarray: Device.CPU, cupy.ndarray: Device.CUDA}


def gpu_available() -> bool:
    """Returns True, if one or more GPUs are available."""
    return Device.CUDA in _AVAILABLE_DEVICES


@cache
def check_device_availability(device: Device):
    """Checks if the specified device is available."""
    if device not in _AVAILABLE_DEVICES:
        raise AttributeError(f"Device {device} is not available.")


@cache
def get_engine(device: _DeviceLike) -> ModuleType:
    """Returns the computation engine for a given device."""
    device = Device(device)
    check_device_availability(device)
    return _DEVICE_TO_ENGINE.get(device, numpy)


@cache
def infer_device(data_type: type) -> Device:
    """Infers the device by type."""
    return _ENGINE_TO_DEVICE.get(data_type, Device.CPU)


def move_data_to_device(data: _ArrayLike, device: Device) -> _ArrayLike:
    """Moves the data to the specified device."""
    if device == infer_device(type(data)):
        return data
    if device == Device.CPU:
        return cupy.asnumpy(data)
    check_device_availability(device)
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
