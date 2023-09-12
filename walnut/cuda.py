"""Cuda functions module"""

import types

import numpy as np
import cupy as cp


__all__ = ["is_availlable"]


def is_availlable() -> bool:
    """Checks if one or more GPUs are availlable.

    Returns
    -------
    bool
        True if one or more GPUs are availlable.
    """
    return cp.is_available()


def get_cpt_pkg(device: str) -> types.ModuleType:
    """Selets a python module for tensor computation for a given device.

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
