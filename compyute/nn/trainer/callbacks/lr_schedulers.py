"""Learning rate scheduler callbacks."""

from typing import Any

from ...optimizers import Optimizer
from ...utils.lr_schedulers import AdaptiveLrScheduler as _AdaptiveLrScheduler
from ...utils.lr_schedulers import CosineLrScheduler as _CosineLrScheduler
from ...utils.lr_schedulers import ExponentialLrScheduler as _ExponentialLrScheduler
from ...utils.lr_schedulers import LrScheduler as _LrScheduler
from ...utils.lr_schedulers import MultistepLrScheduler as _MultistepLrScheduler
from ...utils.lr_schedulers import StepLrScheduler as _StepLrScheduler
from .callback import Callback

__all__ = [
    "StepLrScheduler",
    "MultistepLrScheduler",
    "ExponentialLrScheduler",
    "CosineLrScheduler",
    "AdaptiveLrScheduler",
]


class StepLrScheduler(Callback):
    """Decays an optimizers learning rate at specified time step.

    Parameters
    ----------
    optimizer  : Optimizer
        Optimizer, whose learning rate will be adapted.
    t_decay : int
        Time step, at which the update is applied.
    lr_decay : float, optional
        Decay factor. Defaults to ``0.1``.

    See Also
    --------
    :class:`compyute.nn.utils.lr_schedulers.StepLrScheduler`
    """

    scheduler: _LrScheduler

    def __init__(
        self, optimizer: Optimizer, t_decay: int, lr_decay: float = 0.1
    ) -> None:
        self.scheduler = _StepLrScheduler(optimizer, t_decay, lr_decay)

    def on_step_start(self, trainer_cache: dict[str, Any]) -> None:
        self.scheduler.step()


class MultistepLrScheduler(Callback):
    """Decays an optimizers learning rate each time a specified number of time steps have elapsed.

    Parameters
    ----------
    optimizer: Optimizer
        Optimizer, whose learning rate will be adapted.
    t_decay_step : int
        Time step interval, at which the update is applied.
    lr_decay : float, optional
        Decay factor. Defaults to ``0.1``.

    See Also
    --------
    :class:`compyute.nn.utils.lr_schedulers.MultistepLrScheduler`
    """

    def __init__(
        self, optimizer: Optimizer, t_decay_step: int, lr_decay: float = 0.1
    ) -> None:
        self.scheduler = _MultistepLrScheduler(optimizer, t_decay_step, lr_decay)

    def on_step_start(self, trainer_cache: dict[str, Any]) -> None:
        self.scheduler.step()


class ExponentialLrScheduler(Callback):
    """Decays an optimizers learning rate each time step exponentially.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer, whose learning rate will be adapted.
    decay_steps : int
        How many times the update is applied.
    lr_decay : float, optional
        Decay factor the learning rate is multiplied by each step. Defaults to ``0.1``.

    See Also
    --------
    :class:`compyute.nn.utils.lr_schedulers.ExponentialLrScheduler`
    """

    def __init__(
        self, optimizer: Optimizer, decay_steps: int, lr_decay: float = 0.1
    ) -> None:
        self.scheduler = _ExponentialLrScheduler(optimizer, decay_steps, lr_decay)

    def on_step_start(self, trainer_cache: dict[str, Any]) -> None:
        self.scheduler.step()


class CosineLrScheduler(Callback):
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

    See Also
    --------
    :class:`compyute.nn.utils.lr_schedulers.CosineLrScheduler`
    """

    def __init__(
        self,
        optimizer: Optimizer,
        target_lr: float,
        warmup_steps: int,
        decay_steps: int,
    ) -> None:
        self.scheduler = _CosineLrScheduler(
            optimizer, target_lr, warmup_steps, decay_steps
        )

    def on_step_start(self, trainer_cache: dict[str, Any]) -> None:
        self.scheduler.step()


class AdaptiveLrScheduler(Callback):
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

    See Also
    --------
    :class:`compyute.nn.utils.lr_schedulers.AdaptiveLrScheduler`
    """

    def __init__(
        self,
        optimizer: Optimizer,
        target: str,
        patience: int = 10,
        lr_downscale_factor: float = 0.5,
        lr_upscale_factor: float = 1.0,
    ) -> None:
        self.target = target
        self.scheduler = _AdaptiveLrScheduler(
            optimizer, patience, lr_downscale_factor, lr_upscale_factor
        )

    def on_step_start(self, trainer_cache: dict[str, Any]) -> None:
        if self.target not in trainer_cache:
            raise AttributeError(f"Target {self.target} not found in trainer_cache")
        self.scheduler.step(**{self.target: trainer_cache[self.target]})
