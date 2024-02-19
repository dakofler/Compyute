"""Learning rate schedulers module"""

from abc import ABC, abstractmethod

from compyute.nn.optimizers import Optimizer


__all__ = ["ExpLrScheduler"]


class LrScheduler(ABC):
    """Optimizer base class"""

    def __init__(self, until_epoch: int) -> None:
        self.t: int = 1
        self.optimizer: Optimizer | None = None
        self.until_epoch = until_epoch

    @abstractmethod
    def step(self) -> None:
        """Updates the optimizer learning rate."""


class ExpLrScheduler(LrScheduler):
    """Decays an optimizers learning rate exponentially."""

    def __init__(self, lr_decay: float = 0.5, until_epoch: int = 10) -> None:
        """Decays an optimizers learning rate exponentially.

        Parameters
        ----------
        lr_decay : float, optional
            Decay factor the learning rate is multiplied by each step, by default 0.5.
        until_epoch: int, optional
            Last epoch the update is applied, by default 10.
        """
        super().__init__(until_epoch)
        self.lr_decay = lr_decay

    def step(self) -> None:
        """Decays an optimizers learning rate exponentially."""
        if not self.optimizer:
            raise AttributeError("No optimizer set.")

        if self.t <= self.until_epoch:
            self.optimizer.lr *= self.lr_decay
            self.t += 1
