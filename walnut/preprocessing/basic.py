"""basic preprocessing module"""

import numpy as np
from walnut.tensor import Tensor, AxisLike


def split_train_val_test(
    x: Tensor, ratio_val: float = 0.1, ratio_test: float = 0.1
) -> tuple[Tensor, Tensor, Tensor]:
    """Splits a tensor along axis 0 into three seperate tensor using a given ratio.

    Parameters
    ----------
    x : Tensor
        Tensor to be split.
    ratio_val : float, optional
        Size ratio of the validation split, by default 0.1.
    ratio_test : float, optional
        Size ratio of the test split, by default 0.1.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Train, validation and test tensors.
    """
    shuffle_index = np.arange(len(x.data))
    np.random.shuffle(shuffle_index)
    t_shuffled = x.data[shuffle_index]
    n1 = int(len(t_shuffled) * (1 - ratio_val - ratio_test))
    n2 = int(len(t_shuffled) * (1 - ratio_test))
    train = t_shuffled[:n1]
    val = t_shuffled[n1:n2]
    test = t_shuffled[n2:]
    return Tensor(train), Tensor(val), Tensor(test)


def split_features_labels(x: Tensor, num_x_cols: int) -> tuple[Tensor, Tensor]:
    """Splits a tensor along axis 1 into two seperate tensors.

    Parameters
    ----------
    x : Tensor
        Tensor to be split.
    num_x_cols : int
        Number of feature-colums of the input tensor.

    Returns
    -------
    tuple[Tensor, Tensor]
        First tensor containing features, second tensor containing labels.
    """
    features = Tensor(x.data[:, :num_x_cols])
    labels = Tensor(x.data[:, num_x_cols:])
    return features, labels


def normalize(
    x: Tensor,
    axis: AxisLike | None = None,
    l_bound: int = -1,
    u_bound: int = 1,
) -> Tensor:
    """Normalizes a tensor using min-max feature scaling.

    Parameters
    ----------
    x : Tensor
        Tensor to be normalized.
    axis : AxisLike | None, optional
        Axes over which normalization is applied, by default None.
        If None, the flattended tensor is normalized.
    l_bound : int, optional
        Lower bound of output values, by default -1.
    u_bound : int, optional
        Upper bound of output values, by default 1.

    Returns
    -------
    Tensor
        Normalized tensor.
    """

    x_min = x.min(axis=axis)
    x_max = x.max(axis=axis)
    return (x - x_min) * (u_bound - l_bound) / (x_max - x_min) + l_bound
