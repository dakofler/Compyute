"""Tensorboard integration using `TensorboardX <https://tensorboardx.readthedocs.io/en/latest/tensorboard.html>`_."""

from tensorboardX import SummaryWriter as SR

__all__ = ["SummaryWriter"]


class SummaryWriter(SR):
    """
    SummaryWriter provided by
    `TensorboardX <https://tensorboardx.readthedocs.io/en/latest/tensorboard.html>`_.
    """
