"""Evaluation metrics module"""

from walnut.tensor import Tensor


__all__ = ["accuracy", "r2score"]


def accuracy(logits: Tensor, y_target: Tensor) -> Tensor:
    """Computes the accuracy score.

    Parameters
    ----------
    logits : Tensor
        A model's logits.
    y_target : Tensor
        Target values.

    Returns
    -------
    Tensor
        Accuracy value.
    """

    return (logits.argmax(-1) == y_target).sum() / len(logits)


def r2score(logits: Tensor, y_target: Tensor, eps: float = 1e-8) -> Tensor:
    """Computes the coefficient of determination (R2 score).

    Parameters
    ----------
    logits : Tensor
        A model's logits.
    y_target : Tensor
        Target values.
    eps: float, optional
        Constant for numerical stability, by default 1e-8.

    Returns
    -------
    Tensor
        Accuracy value.
    """

    ssr = ((y_target - logits) ** 2).sum()
    sst = ((y_target - y_target.mean()) ** 2).sum()
    return -ssr / (sst + eps) + 1.0
