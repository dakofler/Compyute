"""Engine functions module"""

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
    return cupy.is_available()


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
    return numpy if device == "cpu" else cupy


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


def set_format():
    """Sets the tensor output to show 4 decimal places."""
    if gpu_available():
        cupy.set_printoptions(
            precision=4, formatter={"float": "{:9.4f}".format}, linewidth=100
        )
    numpy.set_printoptions(
        precision=4, formatter={"float": "{:9.4f}".format}, linewidth=100
    )


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
