"""Engine functions module"""

import os
from types import ModuleType

import cupy
import numpy

from .types import _ArrayLike, _DeviceLike, _ScalarLike

__all__ = ["gpu_available"]


def gpu_available() -> bool:
    """Returns True, if one or more GPUs are available."""
    if "CUDA_PATH" not in os.environ:
        return False
    return bool(cupy.is_available())


def _check_device_availability(device: _DeviceLike):
    """Checks if the specified device is available.

    Raises
    -------
    AttributeError
        If the specified device is not available."""
    if device not in ["cpu"] + (["cuda"] if gpu_available() else []):
        raise AttributeError(f"Device {device} is not available.")


def _get_engine(device: _DeviceLike) -> ModuleType:
    """Selects the computation engine for a given device.

    Parameters
    ----------
    device : _DeviceLike
        Computation device.

    Returns
    -------
    ModuleType
        NumPy or CuPy module.
    """
    _check_device_availability(device)
    return {"cpu": numpy, "cuda": cupy}.get(device, numpy)


def _infer_device(array: _ArrayLike | _ScalarLike) -> _DeviceLike:
    """Infers the device the data is stored on.

    Parameters
    ----------
    array : _ArrayLike | _ScalarLike
        Computation device, options are "cpu" and "cuda". If None, "cpu" is used.

    Returns
    -------
    DeviceLike
        The device the data is stored on."""
    return {numpy.ndarray: "cpu", cupy.ndarray: "cuda"}.get(type(array), "cpu")


def _numpy_to_cupy(numpy_array: numpy.ndarray) -> cupy.ndarray:
    """Converts a NumPy array to a CuPy array.

    Parameters
    ----------
    numpy_array : numpy.ndarray
        NumPy array.

    Returns
    -------
    cupy.ndarray
        CuPy array.
    """
    _check_device_availability("cuda")
    return cupy.array(numpy_array)


def _cupy_to_numpy(cupy_array: cupy.ndarray) -> numpy.ndarray:
    """Converts a CuPy array to a NumPy array.

    Parameters
    ----------
    cupy_array : cupy.ndarray
        CuPy array.

    Returns
    -------
    numpy.ndarray
        NumPy array.
    """
    return cupy.asnumpy(cupy_array)
