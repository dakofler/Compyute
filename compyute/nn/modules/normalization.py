"""Neural network normalization modules."""

from typing import Optional

from ...base_tensor import Tensor, _ShapeLike
from ...dtypes import _DtypeLike, select_dtype_or_float
from ...tensor_functions.creating import ones, zeros
from ..functional.normalizatons import batchnorm1d, batchnorm2d, layernorm
from ..parameter import Buffer, Parameter
from .module import Module

__all__ = ["Batchnorm1d", "Batchnorm2d", "Layernorm"]


class Batchnorm1d(Module):
    r"""Implements Batch Normalization.

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
    dtype : DtypeLike, optional
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.


    .. note::
        Weights are initialized as ones, biases as zeros.
        The running means are initialized as zeros, the running variances as ones.
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        m: float = 0.1,
        dtype: Optional[_DtypeLike] = None,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        super().__init__(label, training)
        self.channels = channels
        self.eps = eps
        self.m = m
        self.dtype = select_dtype_or_float(dtype)

        # parameters
        self.w = Parameter(ones((channels,), self.dtype))
        self.b = Parameter(zeros((channels,), self.dtype))

        # buffers
        self.rmean = Buffer(zeros((channels,), self.dtype))
        self.rvar = Buffer(ones((channels,), self.dtype))

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [2, 3])
        x = x.as_type(self.dtype)

        y, self.rmean, self.rvar, grad_fn = batchnorm1d(
            x, self.rmean, self.rvar, self.w, self.b, self.m, self.eps, self._training
        )

        if self._training and grad_fn is not None:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.as_type(self.dtype)
                dx, dw, db = grad_fn(dy)
                self._update_parameter_grad(self.w, dw)
                self._update_parameter_grad(self.b, db)
                return dx

            self._backward = _backward

        return y


class Batchnorm2d(Module):
    r"""Implements Batch Normalization.

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
    dtype : DtypeLike, optional
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.


    .. note::
        Weights are initialized as ones, biases as zeros.
        The running means are initialized as zeros, the running variances as ones.
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        m: float = 0.1,
        dtype: Optional[_DtypeLike] = None,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        super().__init__(label, training)
        self.channels = channels
        self.eps = eps
        self.m = m
        self.dtype = select_dtype_or_float(dtype)

        # parameters
        self.w = Parameter(ones((channels,), self.dtype))
        self.b = Parameter(zeros((channels,), self.dtype))

        # buffers
        self.rmean = Buffer(zeros((channels,), self.dtype))
        self.rvar = Buffer(ones((channels,), self.dtype))

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [4])
        x = x.as_type(self.dtype)

        y, self.rmean, self.rvar, grad_fn = batchnorm2d(
            x, self.rmean, self.rvar, self.w, self.b, self.m, self.eps, self._training
        )

        if self._training and grad_fn is not None:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.as_type(self.dtype)
                dx, dw, db = grad_fn(dy)
                self._update_parameter_grad(self.w, dw)
                self._update_parameter_grad(self.b, db)
                return dx

            self._backward = _backward

        return y


class Layernorm(Module):
    r"""Implements Layer Normalization.

    .. math::
        y = w \cdot \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} + b

    where :math:`E[x]` and :math:`Var[x]` are computed over all non-batch axes.

    Shapes:
        - Input :math:`(B, ...)`
        - Output :math:`(B, ...)`
    where
        - :math:`B` ... batch axis

    Parameters
    ----------
    normalized_shape : ShapeLike
        Shape of the normalized tensor ignoring the batch dimension.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-5``.
    dtype : DtypeLike, optional
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.


    .. note::
        Weights are initialized as ones, biases as zeros.
    """

    def __init__(
        self,
        normalized_shape: _ShapeLike,
        eps: float = 1e-5,
        dtype: Optional[_DtypeLike] = None,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        super().__init__(label, training)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.dtype = select_dtype_or_float(dtype)

        # parameters
        self.w = Parameter(ones(normalized_shape, self.dtype))
        self.b = Parameter(zeros(normalized_shape, self.dtype))

    def forward(self, x: Tensor) -> Tensor:
        x = x.as_type(self.dtype)

        y, grad_fn = layernorm(x, self.w, self.b, self.eps, self._training)

        if self._training and grad_fn is not None:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.as_type(self.dtype)
                dx, dw, db = grad_fn(dy)
                self._update_parameter_grad(self.w, dw)
                self._update_parameter_grad(self.b, db)
                return dx

            self._backward = _backward

        return y
