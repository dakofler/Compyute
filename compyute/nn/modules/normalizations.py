"""Neural network normalization modules."""

from typing import Optional

from ...tensor_ops.creation_ops import ones, zeros
from ...tensors import ShapeLike, Tensor
from ...typing import DType
from ..functional.normalization_funcs import (
    BatchNorm1DFn,
    BatchNorm2DFn,
    LayerNormFn,
    RMSNormFn,
)
from ..parameter import Buffer, Parameter, update_parameter_grad
from .module import Module

__all__ = ["BatchNorm1D", "BatchNorm2D", "LayerNorm", "RMSNorm"]


class BatchNorm1D(Module):
    r"""Implements Batch Normalization as described by
    `Ioffe et al., 2015 <https://arxiv.org/pdf/1502.03167>`_.

    .. math::
        y = w \cdot \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} + b

    where :math:`E[x]` and :math:`Var[x]` are computed over the
    :math:`B` and :math:`S` axes.

    Shapes:
        - Input :math:`(B, C, S)` or :math:`(B, C)`
        - Output :math:`(B, C, S)` or :math:`(B, C)`
    where
        - :math:`B` ... batch dimension
        - :math:`C` ... channels
        - :math:`S` ... sequence

    Parameters
    ----------
    channels : int
        Number of channels.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-5``.
    m : float, optional
        Momentum used for running mean and variance computation. Defaults to ``0.1``.
    dtype : DType, optional
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights are initialized as ones, biases as zeros.
        The running means are initialized as zeros, the running variances as ones.
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        m: float = 0.1,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.channels = channels
        self.eps = eps
        self.m = m

        # init parameters and buffers
        self.w = Parameter(ones((channels,), dtype=dtype))
        self.b = Parameter(zeros((channels,), dtype=dtype))
        self.rmean = Buffer(zeros((channels,), dtype=dtype))
        self.rvar = Buffer(ones((channels,), dtype=dtype))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        y, rmean, rvar = BatchNorm1DFn.forward(
            self.fcache,
            x,
            self.rmean,
            self.rvar,
            self.w,
            self.b,
            self.m,
            self.eps,
            self._is_training,
        )
        self.rmean.data = rmean.data
        self.rvar.data = rvar.data
        return y

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw, db = BatchNorm1DFn.backward(self.fcache, dy)
        update_parameter_grad(self.w, dw)
        update_parameter_grad(self.b, db)
        return dx


class BatchNorm2D(Module):
    r"""Implements Batch Normalization as described by
    `Ioffe et al., 2015 <https://arxiv.org/pdf/1502.03167>`_.

    .. math::
        y = w \cdot \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} + b

    where :math:`E[x]` and :math:`Var[x]` are computed over the
    :math:`B`, :math:`Y` and :math:`X` axes.

    Shapes:
        - Input :math:`(B, C, Y, X)`
        - Output :math:`(B, C, Y, X)`
    where
        - :math:`B` ... batch dimension
        - :math:`C` ... channels
        - :math:`Y` ... height
        - :math:`X` ... width

    Parameters
    ----------
    channels : int
        Number of channels.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-5``.
    m : float, optional
        Momentum used for running mean and variance computation. Defaults to ``0.1``.
    dtype : DType, optional
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights are initialized as ones, biases as zeros.
        The running means are initialized as zeros, the running variances as ones.
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        m: float = 0.1,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.channels = channels
        self.eps = eps
        self.m = m

        # init parameters and buffers
        self.w = Parameter(ones((channels,), dtype=dtype))
        self.b = Parameter(zeros((channels,), dtype=dtype))
        self.rmean = Buffer(zeros((channels,), dtype=dtype))
        self.rvar = Buffer(ones((channels,), dtype=dtype))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        y, rmean, rvar = BatchNorm2DFn.forward(
            self.fcache,
            x,
            self.rmean,
            self.rvar,
            self.w,
            self.b,
            self.m,
            self.eps,
            self._is_training,
        )
        self.rmean.data = rmean.data
        self.rvar.data = rvar.data
        return y

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw, db = BatchNorm2DFn.backward(self.fcache, dy)
        update_parameter_grad(self.w, dw)
        update_parameter_grad(self.b, db)
        return dx


class LayerNorm(Module):
    r"""Implements Layer Normalization as described by
    `Ba et al., 2016 <https://arxiv.org/pdf/1607.06450>`_.

    .. math::
        y = w \cdot \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} + b

    where :math:`E[x]` and :math:`Var[x]` are computed over feature axes
    specified by `normalized_shape`.

    Shapes:
        - Input :math:`(B, ...)`
        - Output :math:`(B, ...)`
    where
        - :math:`B` ... batch dimension

    Parameters
    ----------
    normalized_shape : _ShapeLike
        Shape of the normalized tensor.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-5``.
    dtype : DType, optional
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights are initialized as ones, biases as zeros.
    """

    def __init__(
        self,
        normalized_shape: ShapeLike,
        eps: float = 1e-5,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.normalized_shape = normalized_shape
        self.eps = eps

        # init parameters
        self.w = Parameter(ones(normalized_shape, dtype=dtype))
        self.b = Parameter(zeros(normalized_shape, dtype=dtype))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return LayerNormFn.forward(self.fcache, x, self.w, self.b, self.eps)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw, db = LayerNormFn.backward(self.fcache, dy)
        update_parameter_grad(self.w, dw)
        update_parameter_grad(self.b, db)
        return dx


class RMSNorm(Module):
    r"""Implements Root Mean Square Layer Normalization as described by
    `Zhang et al., 2019 <https://arxiv.org/pdf/1910.07467>`_.

    .. math::
        y = w \cdot \frac{x}{\text{RMS}(x)} + b

    where :math:`\text{RMS}(x) = \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2}`.
    The :math:`\text{RMS}` is computed over feature axes specified by `normalized_shape`.

    Shapes:
        - Input :math:`(B, ...)`
        - Output :math:`(B, ...)`
    where
        - :math:`B` ... batch dimension

    Parameters
    ----------
    normalized_shape : _ShapeLike
        Shape of the normalized tensor.
    dtype : DType, optional
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights are initialized as ones.
    """

    def __init__(
        self,
        normalized_shape: ShapeLike,
        eps: float = 1e-5,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.normalized_shape = normalized_shape
        self.eps = eps

        # init parameters
        self.w = Parameter(ones(normalized_shape, dtype=dtype))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return RMSNormFn.forward(self.fcache, x, self.w, self.eps)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw = RMSNormFn.backward(self.fcache, dy)
        update_parameter_grad(self.w, dw)
        return dx
