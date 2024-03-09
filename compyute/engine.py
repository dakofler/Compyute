"""Engine functions module"""

import os
import types

import numpy
import cupy


__all__ = ["gpu_available", "set_seed"]

ArrayLike = numpy.ndarray | cupy.ndarray
ScalarLike = (
    numpy.float16
    | numpy.float32
    | numpy.float64
    | numpy.int32
    | numpy.int64
    | cupy.float16
    | cupy.float32
    | cupy.float64
    | cupy.int32
    | cupy.int64
    | list
    | float
    | int
)


def gpu_available() -> bool:
    """Checks if one or more GPUs are available.

    Returns
    -------
    bool
        True if one or more GPUs are available.
    """
    if "CUDA_PATH" not in os.environ:
        return False

    return cupy.is_available()


# create list of available devices
devices = ["cpu"]
if gpu_available():
    devices.append("cuda")
d = ", ".join(devices)
print(f"Compyute: found devices {d}")

# set array output format
formatter = {"float": "{:9.4f}".format}
if gpu_available():
    cupy.set_printoptions(precision=4, formatter=formatter, linewidth=100)
else:
    numpy.set_printoptions(precision=4, formatter=formatter, linewidth=100)


def get_engine(device: str) -> types.ModuleType:
    """Selects the computation engine for a given device.

    Parameters
    ----------
    device : str
        Computation device, options are "cpu" and "cuda".

    Returns
    -------
    types.ModuleType
        NumPy or CuPy module.
    """
    return cupy if device == "cuda" and device in devices else numpy


def set_seed(seed: int) -> None:
    """Sets the seed for RNG for reproducability.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    if gpu_available():
        cupy.random.seed(seed)
    numpy.random.seed(seed)


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
