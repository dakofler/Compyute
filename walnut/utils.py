"""Utility functions module"""

import time
import psutil
import numpy as np


__all__ = ["stopwatch", "memlog", "set_numpy_format"]


def stopwatch(func):
    """Decorator that reports the execution time."""

    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__qualname__, end - start)
        return result

    return wrap


def memlog(func):
    """Decorator that reports the current RAM usage."""

    def wrap(*args, **kwargs):
        result = func(*args, **kwargs)
        print(func.__qualname__, psutil.virtual_memory()[2], "%")
        return result

    return wrap


def set_numpy_format():
    """Sets numpy's float output to show 4 decimal places."""
    np.set_printoptions(
        precision=4, formatter={"float": "{:9.4f}".format}, linewidth=100
    )
