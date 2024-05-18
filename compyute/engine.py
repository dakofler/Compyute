"""Engine functions module"""

import os
from types import ModuleType

import cupy
import numpy

from ._types import _ArrayLike, _DeviceLike, _ScalarLike

__all__ = ["gpu_available"]


def gpu_available() -> bool:
    """Returns True, if one or more GPUs are available."""
    if "CUDA_PATH" not in os.environ:
        return False
    return cupy.is_available()


# create list of available devices
available_devices = ["cpu"] + ["cuda"] if gpu_available() else []
print(f"Compyute: available devices {available_devices}")


def _check_device_availability(device: _DeviceLike):
    """Checks if the specified device is available.

    Raises
    -------
    AttributeError
        If the specified device is not available."""
    if device not in available_devices:
        raise AttributeError(f"Device {device} is not available.")


DEVICE_ENGINES = {"cpu": numpy, "cuda": cupy}


def _get_engine(device: _DeviceLike) -> ModuleType:
    """Selects the computation engine for a given device.

    Parameters
    ----------
    device : DeviceLike | None, optional
        Computation device, options are "cpu" and "cuda". If None, "cpu" is used.

    Returns
    -------
    ModuleType
        NumPy or CuPy module.
    """
    _check_device_availability(device)
    return DEVICE_ENGINES[device]


def _infer_device(data: _ArrayLike | _ScalarLike) -> _DeviceLike:
    """Infers the device the data is stored on.

    Returns
    -------
    DeviceLike
        The device the data is stored on."""
    if isinstance(data, cupy.ndarray):
        return "cuda"
    return "cpu"


def _numpy_to_cupy(np_array: numpy.ndarray) -> cupy.ndarray:
    """Converts a NumPy array to a CuPy array.

    Parameters
    ----------
    np_array : numpy.ndarray
        NumPy array.

    Returns
    -------
    cupy.ndarray
        CuPy array.
    """
    _check_device_availability("cuda")
    return cupy.array(np_array)


def _cupy_to_numpy(cp_array: cupy.ndarray) -> numpy.ndarray:
    """Converts a CuPy array to a NumPy array.

    Parameters
    ----------
    cp_array : cupy.ndarray
        CuPy array.

    Returns
    -------
    numpy.ndarray
        NumPy array.
    """
    return cupy.asnumpy(cp_array)
