"""Training callback module"""

from abc import ABC
from typing import Any

__all__ = ["Callback"]


class Callback(ABC):
    """Trainig callback."""

    __slots__ = ()

    def on_init(self, trainer_cache: dict[str, Any]) -> None:
        """Does someting at initialization."""

    def on_step(self, trainer_cache: dict[str, Any]) -> None:
        """Does someting after each step."""

    def on_epoch_start(self, trainer_cache: dict[str, Any]) -> None:
        """Does someting at the start of each epoch."""

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        """Does someting at the end of each epoch."""
