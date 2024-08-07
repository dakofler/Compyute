"""Learning rate scheduler callbacks."""

from typing import Any

from ...optimizers import Optimizer
from ...utils.lr_schedulers import AdaptiveLrScheduler as _AdaptiveLrScheduler
from ...utils.lr_schedulers import CosineLrScheduler as _CosineLrScheduler
from ...utils.lr_schedulers import ExponentialLrScheduler as _ExponentialLrScheduler
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
    """Decays an optimizers learning rate after a specified epoch.

    Parameters
    ----------
    optimizer  : Optimizer
        Optimizer, whose learning rate will be adapted.
    lr_decay : float, optional
        Decay factor. Defaults to ``0.1``.
    t_decay : int, optional
        Epoch, at which the update is applied. Defaults to ``10``.

    See Also
    --------
    :class:`compyute.nn.utils.lr_schedulers.StepLrScheduler`
    """

    def __init__(self, optimizer: Optimizer, lr_decay: float = 0.1, t_decay: int = 10) -> None:
        self.scheduler = _StepLrScheduler(optimizer, lr_decay, t_decay)

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        self.scheduler.step()


class MultistepLrScheduler(Callback):
    """Decays an optimizers learning rate each time a specified number of epochs has elapsed.

    Parameters
    ----------
    optimizer: Optimizer
        Optimizer, whose learning rate will be adapted.
    lr_decay : float, optional
        Decay factor. Defaults to ``0.5``.
    t_decay_step : int, optional
        Number of epochs, at which the update is applied. Defaults to ``10``.

    See Also
    --------
    :class:`compyute.nn.utils.lr_schedulers.MultistepLrScheduler`
    """

    def __init__(self, optimizer: Optimizer, lr_decay: float = 0.1, t_decay_step: int = 10) -> None:
        self.scheduler = _MultistepLrScheduler(optimizer, lr_decay, t_decay_step)

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        self.scheduler.step()


class ExponentialLrScheduler(Callback):
    """Decays an optimizers learning rate each epoch exponentially.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer, whose learning rate will be adapted.
    lr_decay : float, optional
        Decay factor the learning rate is multiplied by each step. Defaults to ``0.5``.
    decay_steps : int, optional
        How many times (epochs) the update is applied. Defaults to ``10``.

    See Also
    --------
    :class:`compyute.nn.utils.lr_schedulers.ExponentialLrScheduler`
    """

    def __init__(self, optimizer: Optimizer, lr_decay: float = 0.5, decay_steps: int = 10) -> None:
        self.scheduler = _ExponentialLrScheduler(optimizer, lr_decay, decay_steps)

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        self.scheduler.step()


class CosineLrScheduler(Callback):
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

    See Also
    --------
    :class:`compyute.nn.utils.lr_schedulers.CosineLrScheduler`
    """

    def __init__(
        self,
        optimizer: Optimizer,
        target_lr: float = 0.1,
        decay_steps: int = 10,
        warmup_steps: int = 0,
        max_warmup_lr: float = 1.0,
    ) -> None:
        self.scheduler = _CosineLrScheduler(
            optimizer, target_lr, decay_steps, warmup_steps, max_warmup_lr
        )

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        self.scheduler.step()


class AdaptiveLrScheduler(Callback):
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

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        if self.target not in trainer_cache:
            raise AttributeError(f"Target {self.target} not found in trainer_cache")
        self.scheduler.step(**{self.target: trainer_cache[self.target]})
