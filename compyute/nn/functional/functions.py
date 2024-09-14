"""Neural network function"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Any

__all__ = ["Function", "FunctionCache"]


class FunctionCache(dict):
    """LiFo Cache for intermediate data that need
    to be cached for the backward pass."""

    def __getattr__(self, key) -> Any:
        val = self.get(key)
        if not val:  # value is None or empty list
            return None
        return_val = val.pop()
        if not val:  # value is now an empty list
            del self[key]
        return return_val

    def __setattr__(self, key, value) -> None:
        if key not in self:
            self[key] = deque()
        self[key].append(value)

    def __delattr__(self, key) -> None:
        super().__delitem__(key)


class PseudoCache(FunctionCache):
    """Pseudo cache as a placeholder."""

    def __setattr__(self, *args, **kwargs) -> None: ...


class Function(ABC):
    """Neural network function base class."""

    def __init__(self) -> None:
        raise NotImplementedError("Function base class cannot be instantiated.")

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
