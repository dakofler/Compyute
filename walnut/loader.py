"""Loader functions module"""


import pickle
from walnut.tensor import Tensor
from walnut.nn.models import Model


__all__ = ["save", "load"]


def save(obj: Tensor | Model, filename: str = "object") -> None:
    """_summary_

    Parameters
    ----------
    obj : Tensor | Model
        _description_
    filename : str, optional
        _description_, by default "object"
    """
    if isinstance(obj, Model) and obj.optimizer:
        obj.optimizer.reset_grads()
        obj.optimizer.delete_temp_params()
        obj.loss_fn.backward = None
        obj.clean()

    file = open(filename, "wb")
    pickle.dump(obj, file)
    file.close()


def load(filename: str) -> Tensor | Model:
    """_summary_

    Parameters
    ----------
    filename : str
        _description_

    Returns
    -------
    Tensor | Model
        _description_
    """
    file = open(filename, "rb")
    obj = pickle.load(file)
    file.close()
    return obj
