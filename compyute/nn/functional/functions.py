"""Neural network function"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Any

__all__ = ["Function", "FunctionCache"]


class FunctionCache(dict[str, deque]):
    """LiFo Cache for intermediate data that need
    to be cached for the backward pass."""

    def __getattr__(self, key) -> Any:
        item = self.get(key)
        if not item:
            raise KeyError(f"No such item in cache: {key}.")
        value = item.pop()
        if not item:
            del self[key]
        return value

    def __setattr__(self, key, value) -> None:
        if key in self:
            self[key].append(value)
        else:
            self[key] = deque([value])


class PseudoCache(FunctionCache):
    """Pseudo cache as a placeholder."""

    def __setattr__(self, *args, **kwargs) -> None: ...


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
