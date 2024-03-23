"""Engine functions module"""

import os
from types import ModuleType
import numpy
import cupy
from .types import ArrayLike, DeviceLike, ScalarLike


__all__ = ["gpu_available"]


def gpu_available() -> bool:
    """Returns True, if one or more GPUs are available."""
    if "CUDA_PATH" not in os.environ:
        return False
    return cupy.is_available()


# create list of available devices
available_devices = ["cpu"]
if gpu_available():
    available_devices.append("cuda")
print(f"Compyute: found devices {available_devices}")


def check_device(device: DeviceLike):
    """Checks if the specified device is available.

    Raises
    -------
    AttributeError
        If the specified device is not available."""
    if device not in available_devices:
        raise AttributeError(f"Device {device} is not available.")


def get_engine(device: DeviceLike) -> ModuleType:
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
    return cupy if device == "cuda" else numpy


def infer_device(data: ArrayLike | ScalarLike) -> DeviceLike:
    """Infers the device the data is stored on.

    Returns
    -------
    DeviceLike
        The device the data is stored on."""
    if isinstance(data, cupy.ndarray):
        return "cuda"
    return "cpu"


def numpy_to_cupy(np_array: numpy.ndarray) -> cupy.ndarray:
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
    return cupy.array(np_array)


def cupy_to_numpy(cp_array: cupy.ndarray) -> numpy.ndarray:
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
