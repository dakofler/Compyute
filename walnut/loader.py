"""Loader functions module"""


import pickle
from walnut.tensor import Tensor
from walnut.nn.models import Model


__all__ = ["save", "load"]


def save(obj: Tensor | Model, filename: str) -> None:
    """Saves a model or tensor as a binary file.

    Parameters
    ----------
    obj : Tensor | Model
        Object to be saved.
    filename : str
        Name of the file.
    """
    if isinstance(obj, Model) and obj.optimizer:
        obj.optimizer.reset_grads()
        obj.optimizer.reset_temp_params()
        obj.loss_fn.backward = None
        obj.clean()

    file = open(filename, "wb")
    pickle.dump(obj, file)
    file.close()


def load(filename: str) -> Tensor | Model:
    """Load an object from a previously saved binary file.

    Parameters
    ----------
    filename : str
        Name of the saved file.

    Returns
    -------
    Tensor | Model
        Loaded object.
    """
    file = open(filename, "rb")
    obj = pickle.load(file)
    file.close()
    return obj
