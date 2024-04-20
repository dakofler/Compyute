"""Engine functions module"""

import os as __os
from types import ModuleType as __ModuleType
import numpy as __numpy
import cupy as __cupy
from .types import ArrayLike, DeviceLike, ScalarLike


__all__ = ["gpu_available"]


def gpu_available() -> bool:
    """Returns True, if one or more GPUs are available."""
    if "CUDA_PATH" not in __os.environ:
        return False
    return __cupy.is_available()


# create list of available devices
available_devices = ["cpu"] + ["cuda"] if gpu_available() else []
print(f"Compyute: available devices {available_devices}")


def check_device(device: DeviceLike):
    """Checks if the specified device is available.

    Raises
    -------
    AttributeError
        If the specified device is not available."""
    if device not in available_devices:
        raise AttributeError(f"Device {device} is not available.")


DEVICE_ENGINES = {"cpu": __numpy, "cuda": __cupy}


def get_engine(device: DeviceLike) -> __ModuleType:
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
    check_device(device)
    return DEVICE_ENGINES.get(device, __numpy)


def infer_device(data: ArrayLike | ScalarLike) -> DeviceLike:
    """Infers the device the data is stored on.

    Returns
    -------
    DeviceLike
        The device the data is stored on."""
    if isinstance(data, __cupy.ndarray):
        return "cuda"
    return "cpu"


def numpy_to_cupy(np_array: __numpy.ndarray) -> __cupy.ndarray:
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
    check_device("cuda")
    return __cupy.array(np_array)


def cupy_to_numpy(cp_array: __cupy.ndarray) -> __numpy.ndarray:
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
    return __cupy.asnumpy(cp_array)
