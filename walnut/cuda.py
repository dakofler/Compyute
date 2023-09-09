"""Cuda functions module"""


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
