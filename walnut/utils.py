"""Utility functions module"""

import numpy as np


__all__ = ["set_numpy_format", "random_seed"]


def set_numpy_format():
    """Sets numpy's float output to show 4 decimal places."""
    np.set_printoptions(
        precision=4, formatter={"float": "{:9.4f}".format}, linewidth=100
    )


def random_seed(seed: int):
    """Sets the seed for RNG for reproducability.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    np.random.seed(seed)
