"""Learning rate decay callbacks module"""

import math
from .callback import Callback


__all__ = ["ExponentialLR", "StepLR", "MultistepLR"]


class LRDecay(Callback):
    """Optimizer base class"""

    def __init__(self) -> None:
        self.state: dict[str, list[dict[int, float]]] = {"lrs": []}


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

    def on_epoch(self, trainer) -> None:
        """Updates the optimizer learning rate."""
        self.state["lrs"].append({trainer.optimizer.t: trainer.optimizer.lr})
        if trainer.optimizer.t == self.decay_epoch:
            trainer.optimizer.lr *= self.lr_decay


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

    def on_epoch(self, trainer) -> None:
        """Updates the optimizer learning rate."""
        self.state["lrs"].append({trainer.optimizer.t: trainer.optimizer.lr})
        if trainer.optimizer.t % self.decay_step_size == 0:
            trainer.optimizer.lr *= self.lr_decay


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

    def on_epoch(self, trainer) -> None:
        """Updates the optimizer learning rate."""
        self.state["lrs"].append({trainer.optimizer.t: trainer.optimizer.lr})
        if trainer.optimizer.t <= self.until_epoch:
            trainer.optimizer.lr *= self.lr_decay


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

    def on_epoch(self, trainer) -> None:
        """Updates the optimizer learning rate."""
        self.state["lrs"].append({trainer.optimizer.t: trainer.optimizer.lr})

        if trainer.optimizer.t == 1:
            self.lr_max = trainer.optimizer.lr

        if trainer.optimizer.t <= self.until_epoch:
            trainer.optimizer.lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + math.cos(trainer.optimizer.t / self.until_epoch * math.pi)
            )
        else:
            trainer.optimizer.lr = self.lr_min
