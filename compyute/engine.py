"""Engine functions module"""

import os
import types
from typing import Literal

import numpy
import cupy


__all__ = ["gpu_available", "set_seed"]

ArrayLike = numpy.ndarray | cupy.ndarray
DtypeLike = (
    numpy.int8
    | numpy.int16
    | numpy.int32
    | numpy.int64
    | numpy.float16
    | numpy.float32
    | numpy.float64
    | numpy.complex64
    | numpy.complex128
    | cupy.int8
    | cupy.int16
    | cupy.int32
    | cupy.int64
    | cupy.int64
    | cupy.float16
    | cupy.float16
    | cupy.float32
    | cupy.float64
    | cupy.complex64
    | cupy.complex128
    | Literal[
        "int",
        "int8",
        "int16",
        "int32",
        "int64",
        "float",
        "float16",
        "float32",
        "float64",
        "complex",
        "complex64",
        "complex128",
    ]
)
ScalarLike = DtypeLike | list | float | int
ShapeLike = tuple[int, ...]
AxisLike = int | tuple[int, ...]
DeviceLike = Literal["cpu", "cuda"]


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
available_devices = ["cpu"]
if gpu_available():
    available_devices.append("cuda")
d = ", ".join(available_devices)
print(f"Compyute: found devices {d}")

# set array output format
formatter = {"float": "{:9.4f}".format}
if gpu_available():
    cupy.set_printoptions(precision=4, formatter=formatter, linewidth=100)
else:
    numpy.set_printoptions(precision=4, formatter=formatter, linewidth=100)


def check_device(device: DeviceLike):
    """Checks if the specified device is available."""
    if device not in available_devices:
        raise AttributeError(f"Device {device} is not available.")


def get_engine(device: DeviceLike) -> types.ModuleType:
    """Selects the computation engine for a given device.

    Parameters
    ----------
    device : DeviceLike | None, optinal
        Computation device, options are "cpu" and "cuda". If None, "cpu" is used.

    Returns
    -------
    types.ModuleType
        NumPy or CuPy module.
    """
    check_device(device)
    return cupy if device == "cuda" else numpy


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


def infer_device(data: ArrayLike | ScalarLike) -> DeviceLike:
    """Infers the device the data is stored on."""
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
