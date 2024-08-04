"""Compyute engine utilities."""

import os
from contextlib import contextmanager
from enum import Enum
from functools import cache
from types import ModuleType
from typing import Any, Literal, Optional, TypeAlias

import cupy
import numpy

__all__ = [
    "cpu",
    "cuda",
    "gpu_available",
    "get_default_device",
    "set_default_device",
    "default_device",
]


class Device(Enum):
    """Compyute device."""

    CPU = "cpu"
    CUDA = "cuda"

    def __repr__(self) -> str:
        return f"Device('{self.value}')"

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


@cache
def assert_device_available(device: Device) -> None:
    """Raises an error if the specified device is not available."""
    if device in _AVAILABLE_DEVICES:
        return
    raise AttributeError(f"Device {device} is not available.")


def set_default_device(device: Optional[_DeviceLike] = None) -> None:
    """Sets the default device for new tensors.

    Parameters
    ----------
    device : _DeviceLike, optional
        The default device for new tensors. Defaults to ``None``.
        If ``None``, the default device will be deleted.
    """
    if device is None:
        del os.environ["COMPYUTE_DEFAULT_DEVICE"]
        return
    d = Device(device)
    assert_device_available(d)
    os.environ["COMPYUTE_DEFAULT_DEVICE"] = d.value


@contextmanager
def default_device(device: _DeviceLike):
    """Context manager to set the default device for new tensors.

    Parameters
    ----------
    device : _DeviceLike
        The default device for new tensors.
    """
    set_default_device(device)
    try:
        yield
    finally:
        set_default_device()


def get_default_device() -> Optional[Device]:
    """Returns the default device for new tensors.

    Returns
    -------
    Device | None
        The default device for new tensors. ``None`` if no default device is set.
    """
    if "COMPYUTE_DEFAULT_DEVICE" in os.environ:
        return Device(os.environ["COMPYUTE_DEFAULT_DEVICE"])
    return None


def infer_device(data: Any) -> Device:
    """Infers the device from the data.

    Parameters
    ----------
    data : Any
        The data to infer the device from.

    Returns
    -------
    Device
        The inferred device.
    """
    if isinstance(data, cupy.ndarray):
        return Device.CUDA
    return Device.CPU


def select_device_or_cpu(device: Optional[_DeviceLike]) -> Device:
    """Chooses a device based on available options.
    - If ``device`` is not ``None``, it is returned.
    - If ``device`` is ``None`` and a default device is set, the default device is returned.
    - If no default device is set, :class:`compyute.cpu` is returned.

    Parameters
    ----------
    device : _DeviceLike | None
        The device to select.

    Returns
    -------
    Device | None
        The selected device.
    """
    if device is not None:
        return Device(device)
    return get_default_device() or Device.CPU


def select_device_or_infer(data: Any, device: Optional[_DeviceLike]) -> Device:
    """Chooses a device based on available options.
    - If ``device`` is not ``None``, it is returned.
    - If ``device`` is ``None`` and a default device is set, the default device is returned.
    - If no default device is set, the device is inferred from the data.

    Parameters
    ----------
    device : _DeviceLike | None
        The device to select.

    Returns
    -------
    Device | None
        The selected device.
    """
    if device is not None:
        return Device(device)
    return get_default_device() or infer_device(data)


def gpu_available() -> bool:
    """Checks, whether one or more GPUs are available.

    Returns
    -------
    bool
        True, if one or more GPUs are available.
    """
    return Device.CUDA in _AVAILABLE_DEVICES


@cache
def get_engine(device: _DeviceLike) -> ModuleType:
    """Returns the computation engine for a given device."""
    device = Device(device)
    assert_device_available(device)
    return _DEVICE_TO_ENGINE[device]


def move_data_to_device(data: _ArrayLike, device: Device) -> _ArrayLike:
    """Moves the data to the specified device."""
    if device == infer_device(data):
        return data

    if device == Device.CPU:
        return cupy.asnumpy(data)

    assert_device_available(device)
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
