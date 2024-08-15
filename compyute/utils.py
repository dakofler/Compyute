"""Compuyte utils."""

import pickle
from typing import Any

__all__ = ["load", "save"]


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
