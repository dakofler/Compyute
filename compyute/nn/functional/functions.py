"""Neural network function"""

from abc import ABC
from collections import OrderedDict, deque
from typing import Any

__all__ = ["Function", "FunctionCache"]


class FunctionCache(OrderedDict):
    """LiFo Cache for intermediate data that need
    to be cached for the backward pass."""

    def __getattr__(self, key) -> Any:
        value = self.get(key)
        if not value:  # value is None or empty list
            return None
        return value.pop()

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
