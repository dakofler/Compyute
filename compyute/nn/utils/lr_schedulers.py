"""Optimizer learning rate schedulers."""

import math
from abc import ABC, abstractmethod
from itertools import pairwise
from typing import NotRequired, TypedDict

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
        self.cache: LrSchedulerCache = {"lr_history": []}

    def _log_lr(self) -> None:
        self.cache["lr_history"].append(self.optimizer.lr)

    @abstractmethod
    def step(self, **kwargs) -> None:
        """Updates the optimizer learning rate."""


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
    """

    def __init__(self, optimizer: Optimizer, lr_decay: float = 0.1, t_decay: int = 10) -> None:
        super().__init__(optimizer)
        self.lr_decay = lr_decay
        self.t_decay = t_decay

    def step(self, **kwargs) -> None:
        self._log_lr()
        if self.optimizer.t - 1 == self.t_decay:
            self.optimizer.lr *= self.lr_decay


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
    """

    def __init__(self, optimizer: Optimizer, lr_decay: float = 0.1, t_decay_step: int = 10) -> None:
        super().__init__(optimizer)
        self.lr_decay = lr_decay
        self.t_decay_step = t_decay_step

    def step(self, **kwargs) -> None:
        self._log_lr()
        if (self.optimizer.t - 1) % self.t_decay_step == 0:
            self.optimizer.lr *= self.lr_decay


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

    def __init__(self, optimizer: Optimizer, lr_decay: float = 0.5, decay_steps: int = 10) -> None:
        super().__init__(optimizer)
        self.lr_decay = lr_decay
        self.decay_steps = decay_steps

    def step(self, **kwargs) -> None:
        self._log_lr()
        if self.optimizer.t - 1 <= self.decay_steps:
            self.optimizer.lr *= self.lr_decay


class CosineLrScheduler(LrScheduler):
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
    """

    def __init__(
        self,
        optimizer: Optimizer,
        target_lr: float = 0.1,
        decay_steps: int = 10,
        warmup_steps: int = 0,
        max_warmup_lr: float = 1.0,
    ) -> None:
        super().__init__(optimizer)
        self.target_lr = target_lr
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.max_warmup_lr = max_warmup_lr if warmup_steps > 0 else optimizer.lr
        self._initial_lr = optimizer.lr
        self._warmup_lr_delta = (self.max_warmup_lr - self._initial_lr) / max(self.warmup_steps, 1)

    def step(self, **kwargs) -> None:
        self._log_lr()

        # warmup phase
        if self.optimizer.t <= self.warmup_steps:
            self.optimizer.lr = self._initial_lr + self._warmup_lr_delta * self.optimizer.t

        # decay phase
        elif self.optimizer.t <= self.decay_steps + self.warmup_steps:
            d = 1 + math.cos(math.pi * (self.optimizer.t - self.warmup_steps) / (self.decay_steps))
            self.optimizer.lr = self.target_lr + 0.5 * (self.max_warmup_lr - self.target_lr) * d

        # final phase
        else:
            self.optimizer.lr = self.target_lr


class AdaptiveLrScheduler(LrScheduler):
    """Sets an optimizers learning based on the trend of a specified metric.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer, whose learning rate will be adapted.
    patience : int, optional
        Number of past epochs to consider when computing the trend. Defaults to ``10``.
    lr_downscale_factor : float, optional
        Factor to scale the learning rate down by if the metric is not improing.
        Defaults to ``0.5``.
    lr_upscale_factor : float, optional
        Factor to scale the learning rate up by if the metric is improing. Defaults to ``1.0``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        patience: int = 10,
        lr_downscale_factor: float = 0.5,
        lr_upscale_factor: float = 1.0,
    ) -> None:
        super().__init__(optimizer)
        self.patience = patience
        self.lr_downscale_factor = lr_downscale_factor
        self.lr_upscale_factor = lr_upscale_factor

    def step(self, **kwargs) -> None:
        if len(kwargs) != 1:
            raise ValueError("Exactly one metric value must be passed as keyword argument.")

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
