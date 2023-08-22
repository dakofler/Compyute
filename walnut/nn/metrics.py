"""Evaluation metrics module"""

from walnut.tensor import Tensor
from walnut.nn.funcional import softmax


__all__ = ["get_accuracy"]


def get_accuracy(x: Tensor, y: Tensor) -> float:
    """Computes the accuracy score of a prediction compared to target values.

    Parameters
    ----------
    x : Tensor
        A model's logits.
    y : Tensor
        Target values.

    Returns
    -------
    float
        Accuracy value.
    """
    preds = softmax(x).argmax(-1)
    return (preds == y).sum().item() / y.len
