"""Evaluation metrics module"""

from walnut.tensor import Tensor


__all__ = ["get_accuracy"]


def get_accuracy(logits: Tensor, y_true: Tensor) -> Tensor:
    """Computes the accuracy score of a prediction compared to target values.

    Parameters
    ----------
    logits : Tensor
        A model's logits.
    y_true : Tensor
        Target values.

    Returns
    -------
    float
        Accuracy value.
    """

    return (logits.argmax(-1) == y_true).sum() / len(logits)
