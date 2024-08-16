"""Neural network normalization modules."""

from typing import Optional

from ...base_tensor import Tensor, _ShapeLike
from ...dtypes import Dtype, _DtypeLike
from ...tensor_ops.creating import ones, zeros
from ..functional.normalizatons import batchnorm1d, batchnorm2d, layernorm, rmsnorm
from ..parameter import Buffer, Parameter
from .module import Module, validate_input_axes

__all__ = ["BatchNorm1D", "BatchNorm2D", "LayerNorm", "RMSNorm"]


class BatchNorm1D(Module):
    r"""Implements Batch Normalization as described by
    `Ioffe et al., 2015 <https://arxiv.org/pdf/1502.03167>`_.

    .. math::
        y = w \cdot \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} + b

    where :math:`E[x]` and :math:`Var[x]` are computed over the batch axis.

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
    dtype : _DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
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
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.channels = channels
        self.eps = eps
        self.m = m

        # parameters
        self.w = Parameter(ones((channels,), dtype))
        self.b = Parameter(zeros((channels,), dtype))

        # buffers
        self.rmean = Buffer(zeros((channels,), dtype))
        self.rvar = Buffer(ones((channels,), dtype))

    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [2, 3])

        y, self.rmean, self.rvar, grad_fn = batchnorm1d(
            x,
            self.rmean,
            self.rvar,
            self.w,
            self.b,
            self.m,
            self.eps,
            self._is_training,
        )

        if self._is_training and grad_fn is not None:

            def _backward(dy: Tensor) -> Tensor:
                dx, dw, db = grad_fn(dy)
                self._update_parameter_grad(self.w, dw)
                self._update_parameter_grad(self.b, db)
                return dx

            self._backward = _backward

        return y


class BatchNorm2D(Module):
    r"""Implements Batch Normalization as described by
    `Ioffe et al., 2015 <https://arxiv.org/pdf/1502.03167>`_.

    .. math::
        y = w \cdot \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} + b

    where :math:`E[x]` and :math:`Var[x]` are computed over the batch axis.

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
    dtype : _DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
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
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.channels = channels
        self.eps = eps
        self.m = m

        # parameters
        self.w = Parameter(ones((channels,), dtype))
        self.b = Parameter(zeros((channels,), dtype))

        # buffers
        self.rmean = Buffer(zeros((channels,), dtype))
        self.rvar = Buffer(ones((channels,), dtype))

    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [4])

        y, self.rmean, self.rvar, grad_fn = batchnorm2d(
            x,
            self.rmean,
            self.rvar,
            self.w,
            self.b,
            self.m,
            self.eps,
            self._is_training,
        )

        if self._is_training and grad_fn is not None:

            def _backward(dy: Tensor) -> Tensor:
                dx, dw, db = grad_fn(dy)
                self._update_parameter_grad(self.w, dw)
                self._update_parameter_grad(self.b, db)
                return dx

            self._backward = _backward

        return y


class LayerNorm(Module):
    r"""Implements Layer Normalization as described by
    `Ba et al., 2016 <https://arxiv.org/pdf/1607.06450>`_.

    .. math::
        y = w \cdot \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} + b

    where :math:`E[x]` and :math:`Var[x]` are computed over feature axes.

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
    dtype : _DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights are initialized as ones, biases as zeros.
    """

    def __init__(
        self,
        normalized_shape: _ShapeLike,
        eps: float = 1e-5,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.normalized_shape = normalized_shape
        self.eps = eps

        # parameters
        self.w = Parameter(ones(normalized_shape, dtype))
        self.b = Parameter(zeros(normalized_shape, dtype))

    def forward(self, x: Tensor) -> Tensor:

        y, grad_fn = layernorm(x, self.w, self.b, self.eps, self._is_training)

        if self._is_training and grad_fn is not None:

            def _backward(dy: Tensor) -> Tensor:
                dx, dw, db = grad_fn(dy)
                self._update_parameter_grad(self.w, dw)
                self._update_parameter_grad(self.b, db)
                return dx

            self._backward = _backward

        return y


class RMSNorm(Module):
    r"""Implements Root Mean Square Layer Normalization as described by
    `Zhang et al., 2019 <https://arxiv.org/pdf/1910.07467>`_.

    .. math::
        y = w \cdot \frac{x}{\text{RMS}(x)} + b

    where :math:`\text{RMS}(x) = \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2}`

    Shapes:
        - Input :math:`(B, ...)`
        - Output :math:`(B, ...)`
    where
        - :math:`B` ... batch axis

    Parameters
    ----------
    normalized_shape : _ShapeLike
        Shape of the normalized tensor.
    dtype : _DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights are initialized as ones.
    """

    def __init__(
        self,
        normalized_shape: _ShapeLike,
        eps: float = 1e-5,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.normalized_shape = normalized_shape
        self.eps = eps

        # parameters
        self.w = Parameter(ones(normalized_shape, dtype))

    def forward(self, x: Tensor) -> Tensor:

        y, grad_fn = rmsnorm(x, self.w, self.eps, self._is_training)

        if self._is_training and grad_fn is not None:

            def _backward(dy: Tensor) -> Tensor:
                dx, dw = grad_fn(dy)
                self._update_parameter_grad(self.w, dw)
                return dx

            self._backward = _backward

        return y
