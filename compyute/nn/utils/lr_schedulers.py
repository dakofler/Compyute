"""Optimizer learning rate schedulers."""

import math
from abc import ABC, abstractmethod
from itertools import pairwise
from typing import Any, NotRequired, TypedDict

from ..optimizers import Optimizer

__all__ = [
    "LrScheduler",
    "AdaptiveLrScheduler",
    "CosineLrScheduler",
    "ExponentialLrScheduler",
    "StepLrScheduler",
    "MultistepLrScheduler",
]


class LrSchedulerCache(TypedDict):
    """Learning rate scheduler cache"""

    lr_history: list[float]
    target_history: NotRequired[list[float]]


class LrScheduler(ABC):
    """Learning rate scheduler base class.

    Parameters
    ----------
    optimizer  : Optimizer
        Optimizer, whose learning rate will be adapted.
    """

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self.cache = LrSchedulerCache(lr_history=[])

    def _log_lr(self) -> None:
        self.cache["lr_history"].append(self.optimizer.lr)

    @abstractmethod
    def step(self, **kwargs: Any) -> None:
        """Updates the optimizer learning rate."""


class StepLrScheduler(LrScheduler):
    """Decays an optimizers learning rate at specified time step.

    Parameters
    ----------
    optimizer  : Optimizer
        Optimizer, whose learning rate will be adapted.
    t_decay : int
        Time step, at which the update is applied.
    lr_decay : float, optional
        Decay factor. Defaults to ``0.1``.
    """

    def __init__(
        self, optimizer: Optimizer, t_decay: int, lr_decay: float = 0.1
    ) -> None:
        super().__init__(optimizer)
        self.t_decay = t_decay
        self.lr_decay = lr_decay

    def step(self, **kwargs: Any) -> None:
        self._log_lr()
        if self.optimizer.t - 1 == self.t_decay:
            self.optimizer.lr *= self.lr_decay


class MultistepLrScheduler(LrScheduler):
    """Decays an optimizers learning rate each time a specified number of time steps have elapsed.

    Parameters
    ----------
    optimizer: Optimizer
        Optimizer, whose learning rate will be adapted.
    t_decay_step : int
        Time step interval, at which the update is applied.
    lr_decay : float, optional
        Decay factor. Defaults to ``0.1``.
    """

    def __init__(
        self, optimizer: Optimizer, t_decay_step: int, lr_decay: float = 0.1
    ) -> None:
        super().__init__(optimizer)
        self.t_decay_step = t_decay_step
        self.lr_decay = lr_decay

    def step(self, **kwargs: Any) -> None:
        self._log_lr()
        if (self.optimizer.t - 1) % self.t_decay_step == 0:
            self.optimizer.lr *= self.lr_decay


class ExponentialLrScheduler(LrScheduler):
    """Decays an optimizers learning rate each time step exponentially.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer, whose learning rate will be adapted.
    decay_steps : int
        How many times the update is applied.
    lr_decay : float, optional
        Decay factor the learning rate is multiplied by each step. Defaults to ``0.1``.
    """

    def __init__(
        self, optimizer: Optimizer, decay_steps: int, lr_decay: float = 0.1
    ) -> None:
        super().__init__(optimizer)
        self.decay_steps = decay_steps
        self.lr_decay = lr_decay

    def step(self, **kwargs: Any) -> None:
        self._log_lr()
        if self.optimizer.t - 1 <= self.decay_steps:
            self.optimizer.lr *= self.lr_decay


class CosineLrScheduler(LrScheduler):
    """Sets an optimizers learning rate using a cosine schedule.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer, whose learning rate will be adapted.
    target_lr : float
        Target learning rate that will be reached after all decay steps.
    warmup_steps : int
        Number of warmup steps. During these steps the learning rate will
        increase linearly from ~0 to the initial optimizer learning rate.
    decay_steps : int
        How many times the update is applied after the warmup.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        target_lr: float,
        warmup_steps: int,
        decay_steps: int,
    ) -> None:
        super().__init__(optimizer)
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self._max_lr = optimizer.lr

    def step(self, **kwargs: Any) -> None:
        self._log_lr()

        # warmup phase
        if self.optimizer.t <= self.warmup_steps:
            self.optimizer.lr = self._max_lr / self.warmup_steps * self.optimizer.t

        # decay phase
        elif self.optimizer.t <= self.decay_steps + self.warmup_steps:
            decay_t = (self.optimizer.t - self.warmup_steps) / self.decay_steps
            decay = 0.5 * (1.0 + math.cos(math.pi * decay_t))
            self.optimizer.lr = self.target_lr + decay * (self._max_lr - self.target_lr)

        # final phase
        else:
            self.optimizer.lr = self.target_lr


class AdaptiveLrScheduler(LrScheduler):
    """Sets an optimizers learning based on the trend of a specified loss metric.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer, whose learning rate will be adapted.
    patience : int, optional
        Number of past time steps to consider when computing the trend. Defaults to ``10``.
    lr_downscale_factor : float, optional
        Factor to scale the learning rate down by if the metric is not improving.
        Defaults to ``0.5``.
    lr_upscale_factor : float, optional
        Factor to scale the learning rate up by if the metric is improving. Defaults to ``1.0``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        patience: int = 10,
        lr_downscale_factor: float = 0.1,
        lr_upscale_factor: float = 2.0,
    ) -> None:
        super().__init__(optimizer)
        self.patience = patience
        self.lr_downscale_factor = lr_downscale_factor
        self.lr_upscale_factor = lr_upscale_factor

    def step(self, **kwargs: Any) -> None:
        if len(kwargs) != 1:
            raise ValueError("Exactly one metric value must be passed as kwarg.")

        self._log_lr()
        metric = next(iter(kwargs.values()))
        self.cache.setdefault("target_history", []).append(metric)

        if self.optimizer.t > self.patience:
            window = self.cache.get("target_history", [])[-self.patience - 1 :]
            trend = sum(j - i for i, j in pairwise(window))
            if trend < 0:
                self.optimizer.lr *= self.lr_upscale_factor  # model is improving
            else:
                self.optimizer.lr *= self.lr_downscale_factor  # model is not improving
