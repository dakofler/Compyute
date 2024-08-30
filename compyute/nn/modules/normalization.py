"""Neural network normalization modules."""

from typing import Optional

from ...tensor_ops.creating import empty
from ...tensors import ShapeLike, Tensor
from ...typing import DType
from ..functional.normalizations import FBatchNorm1D, FBatchNorm2D, FLayerNorm, FRMSNorm
from ..parameter import Buffer, Parameter, update_parameter_grad
from ..utils.initializers import ones, zeros
from .module import Module, validate_input_axes

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
        - :math:`B` ... batch axis
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
        self.w = Parameter(empty((channels,), dtype=dtype))
        self.b = Parameter(empty((channels,), dtype=dtype))
        self.rmean = Buffer(empty((channels,), dtype=dtype))
        self.rvar = Buffer(empty((channels,), dtype=dtype))
        self._init_parameters_and_buffers()

    def _init_parameters_and_buffers(self) -> None:
        ones(self.w, self.rvar)
        zeros(self.b, self.rmean)

    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [2, 3])

        y, self.rmean, self.rvar = FBatchNorm1D.forward(
            self._fcache,
            x,
            self.rmean,
            self.rvar,
            self.w,
            self.b,
            self.m,
            self.eps,
            self._is_training,
        )

        return y

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        dx, dw, db = FBatchNorm1D.backward(self._fcache, dy)
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
        - :math:`B` ... batch axis
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
        self.w = Parameter(empty((channels,), dtype=dtype))
        self.b = Parameter(empty((channels,), dtype=dtype))
        self.rmean = Buffer(empty((channels,), dtype=dtype))
        self.rvar = Buffer(empty((channels,), dtype=dtype))
        self._init_parameters_and_buffers()

    def _init_parameters_and_buffers(self) -> None:
        ones(self.w, self.rvar)
        zeros(self.b, self.rmean)

    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [4])

        y, self.rmean, self.rvar = FBatchNorm2D.forward(
            self._fcache,
            x,
            self.rmean,
            self.rvar,
            self.w,
            self.b,
            self.m,
            self.eps,
            self._is_training,
        )
        return y

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        dx, dw, db = FBatchNorm2D.backward(self._fcache, dy)
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
        - :math:`B` ... batch axis

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
        self.w = Parameter(empty(normalized_shape, dtype=dtype))
        self.b = Parameter(empty(normalized_shape, dtype=dtype))
        self._init_parameters_and_buffers()

    def _init_parameters_and_buffers(self) -> None:
        ones(self.w)
        zeros(self.b)

    def forward(self, x: Tensor) -> Tensor:
        return FLayerNorm.forward(self._fcache, x, self.w, self.b, self.eps)

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        dx, dw, db = FLayerNorm.backward(self._fcache, dy)
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
        - :math:`B` ... batch axis

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
        self.w = Parameter(empty(normalized_shape, dtype=dtype))
        self._init_parameters_and_buffers()

    def _init_parameters_and_buffers(self) -> None:
        ones(self.w)

    def forward(self, x: Tensor) -> Tensor:
        return FRMSNorm.forward(self._fcache, x, self.w, self.eps)

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        dx, dw = FRMSNorm.backward(self._fcache, dy)
        update_parameter_grad(self.w, dw)
        return dx
