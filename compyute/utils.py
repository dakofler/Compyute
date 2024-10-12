"""Compuyte utils."""

import os
import pickle
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

__all__ = ["load", "save", "set_debug_mode", "debug"]


debug_mode: bool = bool(os.environ.get("COMPYUTE_DEBUG", False))


def get_debug_mode() -> bool:
    """Gets the debug mode."""
    return debug_mode


def set_debug_mode(active: bool) -> None:
    """Sets the debug mode of Compyute.
    When active, more details about tensors and modules are printed.

    Parameters
    ----------
    active : bool
        Whether to enable debug mode.
    """
    global debug_mode
    debug_mode = active


@contextmanager
def debug() -> Generator:
    """Context manager to activate debug mode."""
    debug = get_debug_mode()
    set_debug_mode(True)
    try:
        yield
    finally:
        set_debug_mode(debug)


def save(obj: object, filepath: str) -> None:
    """Saves an object to a binary file.

    Parameters
    ----------
    obj : object
        object to be saved.
    filepath : str
        Where to save the file to.
    """
    with open(filepath, "wb") as file:
        pickle.dump(obj, file)


def load(filepath: str) -> Any:
    """Loads an object from a binary file.

    Parameters
    ----------
    filepath : str
        Path to the binary file to load.

    Returns
    -------
    Any
        Loaded object.
    """
    with open(filepath, "rb") as file:
        obj = pickle.load(file)
    return obj
