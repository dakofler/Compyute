"""Learning rate scheduler callbacks."""

import math
from itertools import pairwise
from typing import Any, NotRequired, TypedDict

from ...optimizers import Optimizer
from .callback import Callback

__all__ = [
    "LrScheduler",
    "AdaptiveLrScheduler",
    "CosineAnnealingLrScheduler",
    "ExponentialLrScheduler",
    "StepLrScheduler",
    "MultistepLrScheduler",
]


class LrCacheEntry(TypedDict):
    """Learning rate cache entry"""

    epoch: int
    lr: float


class LrSchedulerCache(TypedDict):
    """Learning rate scheduler cache"""

    t: int
    lrs: list[LrCacheEntry]
    target_history: NotRequired[list[float]]


class LrScheduler(Callback):
    """Learning rate scheduler base class.

    Parameters
    ----------
    optimizer  : Optimizer
        Optimizer, whose learning rate will be adapted.
    verbose : bool, optional
        Whether to print the learning rate. Defaults to ``False``.
    """

    def __init__(self, optimizer: Optimizer, verbose: bool = False) -> None:
        self.optimizer = optimizer
        self.verbose = verbose
        self.t: int = 0
        self.cache: LrSchedulerCache = {"t": 0, "lrs": []}

    def _update_cache(self, trainer_cache: dict[str, Any]) -> None:
        """Stores the current optimizere learning rate and time step."""
        self.t = trainer_cache["t"]
        self.cache["lrs"].append({"epoch": self.t, "lr": self.optimizer.lr})

    def _print_lr(self) -> None:
        if self.verbose:
            print(f"Epoch {self.t}: lr = {self.optimizer.lr}")


class StepLrScheduler(LrScheduler):
    """Decays an optimizers learning rate after a specified epoch.

    Parameters
    ----------
    optimizer  : Optimizer
        Optimizer, whose learning rate will be adapted.
    lr_decay : float, optional
        Decay factor. Defaults to ``0.1``.
    t_decay : int, optional
        Epoch, at which the update is applied. Defaults to ``10``.
    verbose : bool, optional
        Whether to print the learning rate. Defaults to ``False``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_decay: float = 0.1,
        t_decay: int = 10,
        verbose: bool = False,
    ) -> None:
        super().__init__(optimizer, verbose)
        self.lr_decay = lr_decay
        self.t_decay = t_decay

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        self._update_cache(trainer_cache)
        if trainer_cache["t"] == self.t_decay:
            self.optimizer.lr *= self.lr_decay
        self._print_lr()


class MultistepLrScheduler(LrScheduler):
    """Decays an optimizers learning rate each time a specified number of epochs has elapsed.

    Parameters
    ----------
    optimizer: Optimizer
        Optimizer, whose learning rate will be adapted.
    lr_decay : float, optional
        Decay factor. Defaults to ``0.5``.
    t_decay_step : int, optional
        Number of epochs, at which the update is applied. Defaults to ``10``.
    verbose : bool, optional
        Whether to print the learning rate. Defaults to ``False``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_decay: float = 0.1,
        t_decay_step: int = 10,
        verbose: bool = False,
    ) -> None:
        super().__init__(optimizer, verbose)
        self.lr_decay = lr_decay
        self.t_decay_step = t_decay_step

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        self._update_cache(trainer_cache)
        if self.t % self.t_decay_step == 0:
            self.optimizer.lr *= self.lr_decay
        self._print_lr()


class ExponentialLrScheduler(LrScheduler):
    """Decays an optimizers learning rate each epoch exponentially.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer, whose learning rate will be adapted.
    lr_decay : float, optional
        Decay factor the learning rate is multiplied by each step. Defaults to ``0.5``.
    decay_steps : int, optional
        How many times (epochs) the update is applied. Defaults to ``10``.
    verbose : bool, optional
        Whether to print the learning rate. Defaults to ``False``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_decay: float = 0.5,
        decay_steps: int = 10,
        verbose: bool = False,
    ) -> None:
        super().__init__(optimizer, verbose)
        self.lr_decay = lr_decay
        self.decay_steps = decay_steps

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        self._update_cache(trainer_cache)
        if self.t <= self.decay_steps:
            self.optimizer.lr *= self.lr_decay
        self._print_lr()


class CosineAnnealingLrScheduler(LrScheduler):
    """Sets an optimizers learning rate using a cosine schedule.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer, whose learning rate will be adapted.
    target_lr : float, optional
        Target learning rate. Defaults to ``0.1``.
    decay_steps : int, optional
        How many times (epochs) the update is applied. Defaults to ``10``.
    warmup_steps : int, optional
        Number of warmup steps. Defaults to ``5``.
    max_warmup_lr : float, optional
        Maximum learning rate after warmup. Defaults to ``1``.
    verbose : bool, optional
        Whether to print the learning rate. Defaults to ``False``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        target_lr: float = 0.1,
        decay_steps: int = 10,
        warmup_steps: int = 0,
        max_warmup_lr: float = 1.0,
        verbose: bool = False,
    ) -> None:
        super().__init__(optimizer, verbose)
        self.target_lr = target_lr
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.max_warmup_lr = max_warmup_lr if warmup_steps > 0 else optimizer.lr

        self._initial_lr = optimizer.lr
        self._warmup_lr_delta = (self.max_warmup_lr - self._initial_lr) / max(self.warmup_steps, 1)

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        self._update_cache(trainer_cache)

        # warmup phase
        if self.t <= self.warmup_steps:
            self.optimizer.lr = self._initial_lr + self._warmup_lr_delta * self.t

        # decay phase
        elif self.t <= self.decay_steps + self.warmup_steps:
            decay_factor = 1 + math.cos(math.pi * (self.t - self.warmup_steps) / (self.decay_steps))
            self.optimizer.lr = (
                self.target_lr + 0.5 * (self.max_warmup_lr - self.target_lr) * decay_factor
            )

        # final phase
        else:
            self.optimizer.lr = self.target_lr

        self._print_lr()


class AdaptiveLrScheduler(LrScheduler):
    """Sets an optimizers learning based on the trend of a specified metric.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer, whose learning rate will be adapted.
    target : str, optional
        Metric to consider. Defaults to ``loss``.
    scope : int, optional
        Number of past epochs to consider when computing the trend. Defaults to ``10``.
    lr_downscale_factor : float, optional
        Factor to scale the learning rate down by if the metric is not improing.
        Defaults to ``0.5``.
    lr_upscale_factor : float, optional
        Factor to scale the learning rate up by if the metric is improing. Defaults to ``1.0``.
    verbose : bool, optional
        Whether to print the learning rate. Defaults to ``False``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        target: str = "loss",
        scope: int = 10,
        lr_downscale_factor: float = 0.5,
        lr_upscale_factor: float = 1.0,
        verbose: bool = False,
    ) -> None:
        super().__init__(optimizer, verbose)
        self.target = target
        self.scope = scope
        self.lr_downscale_factor = lr_downscale_factor
        self.lr_upscale_factor = lr_upscale_factor
        self.cache["target_history"] = []

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        self._update_cache(trainer_cache)

        # keep target history
        hist = self.cache.get("target_history", [])
        hist.append(trainer_cache[self.target])

        if self.t > self.scope:
            trend = sum(j - i for i, j in pairwise(hist[-self.scope - 1 :]))
            if trend <= 0:
                self.optimizer.lr *= self.lr_upscale_factor  # model is improving
            else:
                self.optimizer.lr *= self.lr_downscale_factor  # model is not improving

        self._print_lr()
