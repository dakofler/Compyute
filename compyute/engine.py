"""Engine functions module"""

import os
from enum import Enum
from types import ModuleType
from typing import Any, Literal, TypeAlias

import cupy
import numpy

__all__ = ["gpu_available"]


def _cuda_available() -> bool:
    if "CUDA_PATH" not in os.environ:
        return False
    return bool(cupy.is_available())


class Device(Enum):
    """Device enum."""

    CPU = "cpu"
    CUDA = "cuda"

    def __repr__(self) -> str:
        return f"compyute.{self.value}"

    def __str__(self) -> str:
        return self.__repr__()


_ArrayLike: TypeAlias = numpy.ndarray | cupy.ndarray
_DeviceLike: TypeAlias = Literal["cpu", "cuda"] | Device
_AVAILABLE_DEVICES = {Device.CPU}.union({Device.CUDA} if _cuda_available() else {})
_DEVICE_TO_ENGINE: dict[Device, ModuleType] = {Device.CPU: numpy, Device.CUDA: cupy}
_ENGINE_TO_DEVICE: dict[_ArrayLike, Device] = {numpy.ndarray: Device.CPU, cupy.ndarray: Device.CUDA}


def gpu_available() -> bool:
    """Returns True, if one or more GPUs are available."""
    return Device.CUDA in _AVAILABLE_DEVICES


def _check_device_availability(device: Device):
    """Checks if the specified device is available."""
    if device not in _AVAILABLE_DEVICES:
        raise AttributeError(f"Device {device} is not available.")


def get_engine(device: _DeviceLike) -> ModuleType:
    """Returns the computation engine for a given device."""
    device = Device(device)
    _check_device_availability(device)
    return _DEVICE_TO_ENGINE.get(device, numpy)


def infer_device(data: Any) -> Device:
    """Infers the device the data is stored on."""
    return _ENGINE_TO_DEVICE.get(type(data), Device.CPU)


def numpy_to_cupy(numpy_array: numpy.ndarray) -> cupy.ndarray:
    """Converts a NumPy array to a CuPy array."""
    _check_device_availability(Device.CUDA)
    return cupy.array(numpy_array)


def cupy_to_numpy(cupy_array: cupy.ndarray) -> numpy.ndarray:
    """Converts a CuPy array to a NumPy array."""
    return cupy.asnumpy(cupy_array)


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
