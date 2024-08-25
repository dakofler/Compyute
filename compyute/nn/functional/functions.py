"""Neural network function"""

from abc import ABC

from ...tensors import Tensor

__all__ = ["Function", "FunctionCache"]


class FunctionCache(dict):
    """Cache for intermediate tensors"""

    def __getattr__(self, *args, **kwargs) -> Tensor:
        return self.get(*args, **kwargs)

    def __setattr__(self, *args, **kwargs) -> None:
        super().__setitem__(*args, **kwargs)

    def __delattr__(self, *args, **kwargs) -> None:
        super().__delitem__(*args, **kwargs)


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
