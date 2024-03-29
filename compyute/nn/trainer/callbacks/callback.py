"""Training callback module"""

from abc import ABC


class Callback(ABC):
    """Trainig callback."""

    def on_step(self, trainer) -> None:
        """Does someting after each step."""

    def on_epoch(self, trainer) -> None:
        """Does someting after each epoch."""
