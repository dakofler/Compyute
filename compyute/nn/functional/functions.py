"""Neural network function"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Any

__all__ = ["Function", "FunctionCache"]


class FunctionCache:
    """LiFo Cache for intermediate data that need
    to be cached for the backward pass."""

    cache: deque[tuple[Any, ...]]

    def __init__(self) -> None:
        self.cache = deque()

    def push(self, *values: Any) -> None:
        """Adds items to the function cache."""
        self.cache.append(values)

    def pop(self) -> tuple[Any, ...]:
        """Adds the topmost items from the function cache."""
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
