"""Tensorboard util."""

from tensorboardX import SummaryWriter as SR

__all__ = ["SummaryWriter"]


class SummaryWriter(SR):
    """
    SummaryWriter provided by
    `TensorboardX <https://tensorboardx.readthedocs.io/en/latest/tensorboard.html>`_.
    """
