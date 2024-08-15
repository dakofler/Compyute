"""Neural network container modules."""

from itertools import accumulate
from typing import Optional

from ...base_tensor import Tensor
from ...tensor_ops.creating import concatenate, split
from ...tensor_ops.transforming import tensorsum
from .module import Module

__all__ = ["ParallelAdd", "ParallelConcat", "ResidualConnection", "Sequential"]


class Sequential(Module):
    """Container that processes modules sequentially.

    .. math::
        y = f_n( ... f_2(f_1(x)) ... )

    Parameters
    ----------
    *modules : Module
        Modules used in the container.
    label : str, optional
        Container label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(self, *modules: Module, label: Optional[str] = None) -> None:
        super().__init__(label)
        self.modules = list(modules)

    def forward(self, x: Tensor) -> Tensor:
        if not self.modules:
            raise NoChildModulesError()

        for module in self.modules:
            x = module(x)

        if self._is_training:

            def _backward(dy: Tensor) -> Tensor:
                for module in reversed(self.modules):
                    dy = module.backward(dy)
                return dy

            self._backward = _backward

        return x


class ParallelConcat(Module):
    """Container that processes modules in parallel and concatenates their outputs.

    .. math::
        y = concatenate(f_1(x), f_2(x), ..., f_n(x))

    Parameters
    ----------
    *modules : Module
        Modules used in the parallel container.
    concat_axis : int, optional
        Axis along which the output of the parallel modules
        shall be concatinated. Defaults to ``-1``.
    label : str, optional
        Container label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(self, *modules: Module, concat_axis: int = -1, label: Optional[str] = None) -> None:
        super().__init__(label)
        self.modules = list(modules)
        self.concat_axis = concat_axis

    def forward(self, x: Tensor) -> Tensor:
        if not self.modules:
            raise NoChildModulesError()

        ys = [m(x) for m in self.modules]
        y = concatenate(ys, axis=self.concat_axis)

        if self._is_training:
            splits = list(accumulate(y.shape[self.concat_axis] for y in ys[:-1]))

            def _backward(dy: Tensor) -> Tensor:
                dy_splits = split(dy, splits=splits, axis=self.concat_axis)
                return tensorsum(m.backward(s) for m, s in zip(self.modules, dy_splits))

            self._backward = _backward

        return y


class ParallelAdd(Module):
    r"""Container that processes modules in parallel and sums their outputs element-wise.

    .. math::
        y = \sum_i f_i(x)

    Parameters
    ----------
    *modules : Module
        Modules used in the container.
    label : str, optional
        Container label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the container and its modules should be in training mode. Defaults to ``False``.
    """

    def __init__(self, *modules: Module, label: Optional[str] = None) -> None:
        super().__init__(label)
        self.modules = list(modules)

    def forward(self, x: Tensor) -> Tensor:
        if not self.modules:
            raise NoChildModulesError()

        y = tensorsum(m(x) for m in self.modules)

        if self._is_training:
            self._backward = lambda dy: tensorsum(m.backward(dy) for m in self.modules)

        return y


class ResidualConnection(Module):
    """Residual container implementing a residual connection around a block of modules.
    Modules in the residual block are processed sequentially.

    .. math::
        y = x + f(x)

    Parameters
    ----------
    *modules : Module
        Modules used in the residual block. They are processed sequentially.
    residual_projection : Module, optional
        Module used as a projection in the residual pathway to achieve matching dimensions.
        Defaults to ``None``. Using a projection within the residual pathway should be avoided,
        and instead projections should be part of the residual block.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(self, *modules: Module, residual_proj: Optional[Module] = None, label: Optional[str] = None) -> None:
        if len(modules) == 0:
            raise NoChildModulesError("Residual container requires at least one module.")
        super().__init__(label)

        self.block = modules[0] if len(modules) == 1 else Sequential(*modules)
        self.residual_proj = residual_proj

        if residual_proj is not None:
            self.modules = [self.block, residual_proj]
        else:
            self.modules = [self.block]

    def forward(self, x: Tensor) -> Tensor:
        y = self.block(x)
        y += x if self.residual_proj is None else self.residual_proj(x)

        if self._is_training:

            def _backward(dy: Tensor) -> Tensor:
                dx = self.block.backward(dy)
                dx += dy if self.residual_proj is None else self.residual_proj.backward(dy)
                return dx

            self._backward = _backward

        return y


class NoChildModulesError(Exception):
    """Exception for empty containers."""

    def __init__(self, message: str = "Container has no modules.") -> None:
        super().__init__(message)
