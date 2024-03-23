"""Training callback module"""

from abc import ABC


class Callback(ABC):
    """Trainig callback."""

    def __call__(self, trainer, is_step: bool = False) -> None:
        if is_step:
            self.step(trainer)
        else:
            self.epoch(trainer)

    def step(self, trainer) -> None:
        """Does someting after each step."""

    def epoch(self, trainer) -> None:
        """Does someting after each epoch."""


class Yell(Callback):
    """Just yells."""

    def epoch(self, trainer) -> None:
        print("AAAAAAAAAHHHHHHHHH!!")


class Swear(Callback):
    """Swears."""

    def step(self, trainer) -> None:
        print("FUCK!!")