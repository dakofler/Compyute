"""Training callback module"""

from typing import Any
from abc import ABC


__all__ = ["Callback"]


class Callback(ABC):
    """Trainig callback."""

    def on_init(self, state: dict[str, Any]) -> None:
        """Does someting at initialization."""

    def on_step(self, state: dict[str, Any]) -> None:
        """Does someting after each step."""

    def on_epoch_start(self, state: dict[str, Any]) -> None:
        """Does someting at the start of each epoch."""

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        """Does someting at the end of each epoch."""
