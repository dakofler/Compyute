"""Neural network function"""

from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from typing import Any

__all__ = ["Function", "FunctionCache"]


class FunctionCache:
    """
    LiFo Cache for intermediate data that need
    to be cached for gradient computation.
    """

    cache: deque[tuple[Any, ...]]

    def __init__(self) -> None:
        self.cache = deque()

    def push(self, *items: Any) -> None:
        """Adds items to the function cache."""
        if get_caching_enabled():
            self.cache.append(items)

    def pop(self) -> tuple[Any, ...]:
        """Removes and returns the topmost items from the function cache."""
        return self.cache.pop()


class PseudoCache(FunctionCache):
    """Pseudo cache as a placeholder."""

    def push(self, *values: Any) -> None: ...


class Function(ABC):
    """Neural network function base class."""

    def __init__(self) -> None:
        raise NotImplementedError("Function cannot be instantiated.")

    def __call__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Use ``forward()`` instead.")

    @staticmethod
    @abstractmethod
    def forward(*args, **kwargs) -> Any:
        """Forward pass of the function."""

    @staticmethod
    @abstractmethod
    def backward(*args, **kwargs) -> Any:
        """Backward pass of the function."""


caching_enabled: bool = True


def get_caching_enabled() -> bool:
    """Returns ``True`` if caching is enabled."""
    return caching_enabled


def set_caching_enabled(enabled: bool) -> None:
    """Sets whether caching of values for gradient computation is enabled."""
    global caching_enabled
    caching_enabled = enabled


@contextmanager
def no_caching():
    """Context manager to disable caching of values for gradient computation."""
    set_caching_enabled(False)
    try:
        yield
    finally:
        set_caching_enabled(True)
