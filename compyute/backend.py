"""Compyute engine utilities."""

from abc import ABC
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional, TypeAlias

import cupy
import numpy

__all__ = ["cpu", "cuda", "Device", "use_device"]


class Device(ABC):
    """Device base class."""

    type: str

    @property
    def module(self) -> ModuleType:
        """Computation engine."""
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        return isinstance(other, Device) and repr(self) == repr(other)

    @property
    def properties(self) -> Optional[dict[str, Any]]:
        """Returns information about the device."""
        ...

    @property
    def memory_info(self) -> Optional[dict[str, int]]:
        """Returns information about the device memory in bytes."""
        ...


class CPU(Device):
    """CPU device."""

    type: str = "cpu"

    def __repr__(self) -> str:
        return 'compyute.Device(type="cpu")'

    @property
    def module(self) -> ModuleType:
        return numpy


class CUDA(Device):
    """GPU device."""

    type: str = "cuda"
    index: int
    _cupy_device: cupy.cuda.Device

    def __init__(self, index: int = 0):
        self.index = index
        self._cupy_device = cupy.cuda.Device(self.index)

    def __repr__(self):
        return f'compyute.Device(type="cuda:{self.index}")'

    def __enter__(self, *args):
        return self._cupy_device.__enter__(*args)

    def __exit__(self, *args):
        return self._cupy_device.__exit__(*args)

    @property
    def module(self) -> ModuleType:
        return cupy

    @property
    def properties(self) -> Optional[dict[str, Any]]:
        return cupy.cuda.runtime.getDeviceProperties(self.index)

    @property
    def memory_info(self) -> dict[str, int]:
        free, total = self._cupy_device.mem_info
        return {"used": total - free, "free": free, "total": total}


cpu = CPU()
cuda = CUDA(0)


ArrayLike: TypeAlias = numpy.ndarray | cupy.ndarray


def data_to_device(data: ArrayLike, device: Device) -> ArrayLike:
    """Moves the data to the specified device."""
    if device == cpu:
        return cupy.asnumpy(data)
    return cupy.asarray(data)


def gpu_available() -> bool:
    """Checks if GPU is available."""
    try:
        return cupy.cuda.is_available()
    except Exception:
        return False


def get_device_count() -> int:
    """Returns the number of available devices."""
    if not gpu_available():
        return 0
    return cupy.cuda.runtime.getDeviceCount()


def free_cuda_memory() -> None:
    """Frees unused blocks from the GPU memory."""
    if not gpu_available():
        return
    cupy.get_default_memory_pool().free_all_blocks()


def synchronize() -> None:
    """Synchronizes devices tot he current thread."""
    if not gpu_available():
        return
    cupy.cuda.runtime.deviceSynchronize()


def show_cuda_config() -> None:
    """Prints the CUDA configuration."""
    if not gpu_available():
        return
    cupy.show_config()


fallback_default_device: Device = cpu
default_device: Optional[Device] = None


def set_default_device(device: Optional[Device]) -> None:
    """Sets the default device."""
    global default_device
    default_device = device


@contextmanager
def use_device(device: Device):
    """Context manager to set the default device when creating tensors."""
    set_default_device(device)
    try:
        yield
    finally:
        set_default_device(None)


from typing import Any


def get_device_from_array(array: ArrayLike) -> Device:
    """Infers the device by type."""
    if isinstance(array, cupy.ndarray):
        return cuda
    return cpu


def get_default_device() -> Optional[Device]:
    """Returns the default device."""
    return default_device


def select_device(device: Optional[Device]) -> Device:
    """Selects the device. Returns the default device if device is ``None``."""
    return device or default_device or fallback_default_device


class DeviceError(Exception):
    """Tensors with mismatching devices."""
