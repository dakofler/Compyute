"""Training callback base class."""

from abc import ABC
from typing import Any

__all__ = ["Callback"]


class Callback(ABC):
    """Trainig callback base class."""

    def on_init(self, trainer_cache: dict[str, Any]) -> None:
        """Does someting at initialization.

        Parameters
        ----------
        trainer_cache : dict[str, Any]
            Trainer cache.
        """

    def on_step(self, trainer_cache: dict[str, Any]) -> None:
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
