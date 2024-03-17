"""Learning rate decay module"""

from abc import ABC
import math
from compyute.nn.optimizers.optimizers import Optimizer


__all__ = ["ExponentialLR", "StepLR", "MultistepLR"]


class LRDecay(ABC):
    """Optimizer base class"""

    def __init__(self) -> None:
        self.t: int = 1
        self.optimizer: Optimizer | None = None

    def step(self) -> None:
        """Updates the optimizer learning rate."""
        if not self.optimizer:
            raise AttributeError("No optimizer set.")


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

    def step(self) -> None:
        """Updates the optimizer learning rate."""
        super().step()
        if self.t == self.decay_epoch:
            self.optimizer.lr *= self.lr_decay
        self.t += 1


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

    def step(self) -> None:
        """Updates the optimizer learning rate."""
        super().step()
        if self.t % self.decay_step_size == 0:
            self.optimizer.lr *= self.lr_decay
        self.t += 1


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

    def step(self) -> None:
        """Updates the optimizer learning rate."""
        super().step()
        if self.t <= self.until_epoch:
            self.optimizer.lr *= self.lr_decay
            self.t += 1


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

    def step(self) -> None:
        """Updates the optimizer learning rate."""
        super().step()

        # set lr_ax to be the initial learning rate
        if self.t == 1:
            self.lr_max = self.optimizer.lr

        if self.t <= self.until_epoch:
            self.optimizer.lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1.0 + math.cos(self.t / self.until_epoch * math.pi)
            )
        else:
            self.optimizer.lr = self.lr_min
        self.t += 1
