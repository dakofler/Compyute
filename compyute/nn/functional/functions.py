"""Neural network function"""

from abc import ABC
from typing import Any

__all__ = ["Function", "FunctionCache"]


class FunctionCache(dict):
    """LiFo Cache for intermediate data that need
    to be cached for the backward pass."""

    def __getattr__(self, key) -> Any:
        value = self.get(key)
        if not value:
            return None
        ret_value = value.pop()
        if not value:
            del self[key]
        return ret_value

    def __setattr__(self, key, value) -> None:
        if key in self:
            self[key].append(value)
        else:
            self[key] = [value]

    def __delattr__(self, key) -> None:
        super().__delitem__(key)


class PseudoCache(FunctionCache):
    """Pseudo cache as a placeholder."""

    def __setattr__(self, *args, **kwargs) -> None:
        pass


class Function(ABC):
    """Neural network function base class."""

    def __init__(self) -> None:
        raise NotImplementedError("Function base class cannot be instantiated.")

    def __call__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Use ``forward()`` instead.")
