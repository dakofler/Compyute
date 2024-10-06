"""Training callback base class."""

from abc import ABC
from enum import Enum, auto
from typing import Any

__all__ = ["Callback"]


class CallbackTrigger(Enum):
    RUN_START = auto()
    RUN_END = auto()
    EPOCH_START = auto()
    EPOCH_END = auto()
    STEP_START = auto()
    STEP_END = auto()


class Callback(ABC):
    """Trainig callback base class."""

    def on_run_start(self, trainer_cache: dict[str, Any]) -> None:
        """Does someting at the start of the training run.

        Parameters
        ----------
        trainer_cache : dict[str, Any]
            Trainer cache.
        """

    def on_step_start(self, trainer_cache: dict[str, Any]) -> None:
        """Does someting before each step.

        Parameters
        ----------
        trainer_cache : dict[str, Any]
            Trainer cache.
        """

    def on_step_end(self, trainer_cache: dict[str, Any]) -> None:
        """Does someting after each step.

        Parameters
        ----------
        trainer_cache : dict[str, Any]
            Trainer cache.
        """

    def on_epoch_start(self, trainer_cache: dict[str, Any]) -> None:
        """Does someting at the start of each epoch.

        Parameters
        ----------
        trainer_cache : dict[str, Any]
            Trainer cache.
        """

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        """Does someting at the end of each epoch.

        Parameters
        ----------
        trainer_cache : dict[str, Any]
            Trainer cache.
        """

    def on_run_end(self, trainer_cache: dict[str, Any]) -> None:
        """Does someting at the end of the training run.

        Parameters
        ----------
        trainer_cache : dict[str, Any]
            Trainer cache.
        """
