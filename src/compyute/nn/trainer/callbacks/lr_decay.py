"""Learning rate decay callbacks module"""

import math
from typing import Any

from ....tensor_functions.computing import tensorsum
from .callback import Callback

__all__ = ["ExponentialLR", "StepLR", "MultistepLR", "AdaptiveLR"]


class LRDecay(Callback):
    """Optimizer base class"""

    def __init__(self) -> None:
        self.cache: dict[str, list[dict]] = {"lrs": []}

    def _log_lr(self, trainer_cache: dict[str, Any]) -> None:
        """Stores the current optimizere learning rate."""
        self.cache["lrs"].append({"epoch": trainer_cache["t"], "lr": trainer_cache["optimizer"].lr})


class StepLR(LRDecay):
    """Decays an optimizers learning rate after a specified epoch."""

    def __init__(self, lr_decay: float = 0.1, decay_epoch: int = 10) -> None:
        """Decays an optimizers learning rate after a specified epoch.

        Parameters
        ----------
        lr_decay : float, optional
            Decay factor, by default 0.1.
        decay_epoch: int, optional
            Epoch, after which the update is applied, by default 10.
        """
        super().__init__()
        self.lr_decay = lr_decay
        self.decay_epoch = decay_epoch

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        self._log_lr(trainer_cache)
        if trainer_cache["t"] == self.decay_epoch:
            trainer_cache["optimizer"].lr *= self.lr_decay


class MultistepLR(LRDecay):
    """Decays an optimizers learning rate each time a specified number of epochs has elapsed."""

    def __init__(self, lr_decay: float = 0.1, decay_step_size: int = 10) -> None:
        """Decays an optimizers learning rate each time a specified number of epochs has elapsed.

        Parameters
        ----------
        lr_decay : float, optional
            Decay factor, by default 0.5.
        decay_step_size: int, optional
            Number of epochs, after which the update is applied, by default 10.
        """
        super().__init__()
        self.lr_decay = lr_decay
        self.decay_step_size = decay_step_size

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        self._log_lr(trainer_cache)
        if trainer_cache["t"] % self.decay_step_size == 0:
            trainer_cache["optimizer"].lr *= self.lr_decay


class ExponentialLR(LRDecay):
    """Decays an optimizers learning rate exponentially."""

    def __init__(self, lr_decay: float = 0.5, until_epoch: int = 10) -> None:
        """Decays an optimizers learning rate each epoch exponentially.

        Parameters
        ----------
        lr_decay : float, optional
            Decay factor the learning rate is multiplied by each step, by default 0.5.
        until_epoch: int, optional
            Last epoch the update is applied, by default 10.
        """
        super().__init__()
        self.lr_decay = lr_decay
        self.until_epoch = until_epoch

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        self._log_lr(trainer_cache)
        if trainer_cache["t"] <= self.until_epoch:
            trainer_cache["optimizer"].lr *= self.lr_decay


class CosineLR(LRDecay):
    """Sets an optimizers learning rate using a cosine schedule."""

    def __init__(self, lr_min: float = 0.0, until_epoch: int = 10) -> None:
        """Sets an optimizers learning rate using a cosine schedule.

        Parameters
        ----------
        lr_min : float, optional
            Minimum learning rate, by default 0.0.
        until_epoch: int, optional
            Last epoch the update is applied, by default 10.
        """
        super().__init__()
        self.lr_min = lr_min
        self.until_epoch = until_epoch
        self.lr_max = 1.0

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        self._log_lr(trainer_cache)

        # set max lr to initial lr
        if trainer_cache["t"] == 1:
            self.lr_max = trainer_cache["optimizer"].lr

        # decay lr
        if trainer_cache["t"] <= self.until_epoch:
            trainer_cache["optimizer"].lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + math.cos(trainer_cache["t"] / self.until_epoch * math.pi)
            )
        else:
            trainer_cache["optimizer"].lr = self.lr_min


class AdaptiveLR(LRDecay):
    """Sets an optimizers learning rate based on the trend of a specified metric."""

    def __init__(
        self,
        target: str = "loss",
        epoch_range: int = 10,
        lr_downscale_factor: float = 0.5,
        lr_upscale_factor: float = 1.0,
    ) -> None:
        """Sets an optimizers learning based on the trend of a specified metric.

        Parameters
        ----------
        target : str, optional
            Metric to consider, by default "loss".
        epoch_range : int, optional
            Number of past epochs to consider, by default 10.
        lr_downscale_factor : float, optional
            Factor to scale the learning rate down by if the metric is not improing, by default 0.5.
        lr_upscale_factor : float, optional
            Factor to scale the learning rate up by if the metric is improing, by default 1.0.
        """
        super().__init__()
        self.target = target
        self.epoch_range = epoch_range
        self.lr_downscale_factor = lr_downscale_factor
        self.lr_upscale_factor = lr_upscale_factor
        self.cache["history"] = []

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        # log lr
        self._log_lr(trainer_cache)

        # keep target history
        h = self.cache["history"]
        h.append(trainer_cache[self.target])

        if trainer_cache["t"] > self.epoch_range:

            # compute target trend
            trend = tensorsum(h[-i] - h[-i - 1] for i in range(1, self.epoch_range + 1))

            if trend <= 0:
                # model is improving
                trainer_cache["optimizer"].lr *= self.lr_upscale_factor
            else:
                # model is not improving
                trainer_cache["optimizer"].lr *= self.lr_downscale_factor
