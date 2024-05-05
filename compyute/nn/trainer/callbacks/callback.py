"""Training callback module"""

from typing import Any
from abc import ABC


class Callback(ABC):
    """Trainig callback."""

    def on_step(self, state: dict[str, Any]) -> None:
        """Does someting after each step."""

    def on_epoch(self, state: dict[str, Any]) -> None:
        """Does someting after each epoch."""
