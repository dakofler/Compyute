"""Compyute engine utilities."""

from abc import ABC
from contextlib import contextmanager
from functools import cache
from typing import Optional, TypeAlias

import cupy
import numpy

__all__ = ["cpu", "cuda", "Device", "use_device"]


class Device(ABC):
    """Device base class."""

    type: str

    @property
    def engine(self):
        """Computation engine."""
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        return isinstance(other, Device) and repr(self) == repr(other)


class CPU(Device):
    """CPU device."""

    type: str = "cpu"

    def __repr__(self):
        return 'compyute.Device(type="cpu")'

    @property
    def engine(self):
        return numpy


class CUDA(Device):
    """GPU device."""

    type: str = "cuda"

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
    def engine(self):
        return cupy


cpu = CPU()
cuda = CUDA(0)


ArrayLike: TypeAlias = numpy.ndarray | cupy.ndarray


def data_to_device(data: ArrayLike, device: Device) -> ArrayLike:
    """Moves the data to the specified device."""
    if device == cpu:
        return cupy.asnumpy(data)
    return cupy.asarray(data)


@cache
def gpu_available() -> bool:
    """Checks if GPU is available."""
    try:
        return cupy.cuda.is_available()
    except Exception:
        return False


@cache
def get_device_count() -> int:
    """Returns the number of available devices."""
    return cupy.cuda.runtime.getDeviceCount()


def get_cuda_memory_usage() -> tuple[int, int]:
    """Returns the amount of GPU memory used."""
    if not gpu_available():
        return 0, 0
    mempool = cupy.get_default_memory_pool()
    used_bytes = mempool.used_bytes()
    total_bytes = mempool.total_bytes()
    return used_bytes, total_bytes


def flush_cuda_memory() -> None:
    """Flushes the GPU memory."""
    if not gpu_available():
        return
    cupy.get_default_memory_pool().free_all_blocks()


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


@cache
def get_device_from_class(array_type: type) -> Device:
    """Infers the device by type."""
    if array_type == cupy.ndarray:
        return cuda
    return cpu


def get_default_device() -> Optional[Device]:
    """Returns the default device."""
    return default_device


def select_device(device: Optional[Device]) -> Device:
    """Selects the device. Returns the default device if device is ``None``."""
    return device or default_device or fallback_default_device
