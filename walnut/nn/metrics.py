"""evaluation metrics module"""

import numpy as np
from walnut import tensor
from walnut.tensor import Tensor


def accuracy(output: Tensor, targets: Tensor) -> float:
    """Computes the accuracy score of a prediction compared to target values."""

    # create tensor with ones where highest probabilities occur
    preds = tensor.zeros_like(output).data
    p_b, _ = preds.shape
    max_prob_indices = np.argmax(output.data, axis=1)
    preds[np.arange(0, p_b), max_prob_indices] = 1

    # count number of correct samples
    num_correct_preds = np.sum(preds * targets.data)
    return num_correct_preds / p_b
