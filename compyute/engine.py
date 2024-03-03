"""Engine functions module"""

import types

import numpy as np
import cupy as cp


__all__ = ["gpu_available", "set_seed"]

ArrayLike = np.ndarray | cp.ndarray
ScalarLike = (
    np.float16
    | np.float32
    | np.float64
    | np.int32
    | np.int64
    | cp.float16
    | cp.float32
    | cp.float64
    | cp.int32
    | cp.int64
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
    return cp.is_available()


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
    return np if device == "cpu" else cp


def set_seed(seed: int) -> None:
    """Sets the seed for RNG for reproducability.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    if gpu_available():
        cp.random.seed(seed)
    np.random.seed(seed)


def set_format():
    """Sets the tensor output to show 4 decimal places."""
    if gpu_available():
        cp.set_printoptions(
            precision=4, formatter={"float": "{:9.4f}".format}, linewidth=100
        )
    np.set_printoptions(
        precision=4, formatter={"float": "{:9.4f}".format}, linewidth=100
    )


def numpy_to_cupy(np_array: np.ndarray) -> cp.ndarray:
    """Converts a NumPy array to a CuPy array.

    Parameters
    ----------
    np_array : np.ndarray
        NumPy array.

    Returns
    -------
    cp.ndarray
        CuPy array.
    """
    return cp.array(np_array)


def cupy_to_numpy(cp_array: cp.ndarray) -> np.ndarray:
    """Converts a CuPy array to a NumPy array.

    Parameters
    ----------
    cp_array : cp.ndarray
        CuPy array.

    Returns
    -------
    np.ndarray
        NumPy array.
    """
    return cp.asnumpy(cp_array)
