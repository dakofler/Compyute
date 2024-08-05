"""Neural network convolution modules."""

from typing import Literal, Optional

from ...base_tensor import Tensor
from ...dtypes import Dtype, _DtypeLike
from ...random.random import uniform
from ...tensor_functions.creating import zeros
from ..functional.convolutions import (
    _PaddingLike,
    avgpooling2d,
    convolve1d,
    convolve2d,
    maxpooling2d,
)
from ..parameter import Parameter
from .module import Module

__all__ = ["Convolution1d", "Convolution2d", "MaxPooling2d", "AvgPooling2d"]


class Convolution1d(Module):
    r"""Applies a 1D convolution to the input for feature extraction.

    .. math::
        y = b + \sum_{k=0}^{C_{in}-1} w_{k}*x_{k}

    where :math:`*` is the cross-correlation operator.

    Shapes:
        - Input :math:`(B, C_{in}, S_{in})`
        - Output :math:`(B, C_{out}, S_{out})`
    where
        - :math:`B` ... batch axis
        - :math:`C_{in}` ... input channels
        - :math:`S_{in}` ... input sequence
        - :math:`C_{out}` ... output channels
        - :math:`S_{out}` ... output sequence

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (filters).
    kernel_size : int
        Size of each kernel.
    padding : _PaddingLike, optional
        Padding applied before convolution. Defaults to ``valid``.
    stride : int, optional
        Stride used for the convolution operation. Defaults to ``1``.
    dilation : int, optional
        Dilation used for each axis of the filter. Defaults to ``1``.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    dtype : DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.


    .. note::
        Weights are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in} * k * k}}`. Biases are initialized as zeros.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Literal["valid", "same"] = "valid",
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        super().__init__(label, training)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.dtype = Dtype(dtype)

        # init weights
        k = (in_channels * kernel_size) ** -0.5
        self.w = Parameter(uniform((out_channels, in_channels, kernel_size), -k, k, dtype))

        # init biases
        self.b = Parameter(zeros((out_channels,), dtype)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [3])
        x = x.to_type(self.dtype)
        y, grad_fn = convolve1d(
            x, self.w, self.b, self.padding, self.stride, self.dilation, self._training
        )

        if self._training and grad_fn is not None:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.to_type(self.dtype)
                dx, dw, db = grad_fn(dy)
                self._update_parameter_grad(self.w, dw)
                self._update_parameter_grad(self.b, db)
                return dx

            self._backward = _backward

        return y


class Convolution2d(Module):
    r"""Applies a 2D convolution to the input for feature extraction.

    .. math::
        y = b + \sum_{k=0}^{C_{in}-1} w_{k}*x_{k}

    where :math:`*` is the cross-correlation operator.

    Shapes:
        - Input :math:`(B, C_{in}, Y_{in}, X_{in})`
        - Output :math:`(B, C_{out}, Y_{out}, X_{out})`
    where
        - :math:`B` ... batch axis
        - :math:`C_{in}` ... input channels
        - :math:`Y_{in}` ... input height
        - :math:`X_{in}` ... input width
        - :math:`C_{out}` ... output channels
        - :math:`Y_{out}` ... output height
        - :math:`X_{out}` ... output width

    Parameters
    ----------
    in_channels : int
        Number of input channels (color channels).
    out_channels : int
        Number of output channels (filters or feature maps).
    kernel_size : int
        Size of each kernel.
    padding : _PaddingLike, optional
        Padding applied before convolution. Defaults to ``valid``.
    stride : int , optional
        Strides used for the convolution operation. Defaults to ``1``.
    dilation : int , optional
        Dilations used for each axis of the filter. Defaults to ``1``.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    dtype : DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.


    .. note::
        Weights are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in} * k * k}}`. Biases are initialized as zeros.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: _PaddingLike = "valid",
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        super().__init__(label, training)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.dtype = Dtype(dtype)

        # init weights
        k = (in_channels * kernel_size**2) ** -0.5
        self.w = Parameter(
            uniform((out_channels, in_channels, kernel_size, kernel_size), -k, k, dtype)
        )

        # init biases
        self.b = Parameter(zeros((out_channels,), dtype)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [4])
        x = x.to_type(self.dtype)
        y, grad_fn = convolve2d(
            x, self.w, self.b, self.padding, self.stride, self.dilation, self._training
        )

        if self._training and grad_fn is not None:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.to_type(self.dtype)
                dx, dw, db = grad_fn(dy)
                self._update_parameter_grad(self.w, dw)
                self._update_parameter_grad(self.b, db)
                return dx

            self._backward = _backward

        return y


class MaxPooling2d(Module):
    """MaxPooling layer used for downsampling.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the pooling window used for the pooling operation. Defaults to ``2``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.
    """

    def __init__(
        self, kernel_size: int = 2, label: Optional[str] = None, training: bool = False
    ) -> None:
        super().__init__(label, training)
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [4])

        kernel_size = (self.kernel_size, self.kernel_size)
        y, self._backward = maxpooling2d(x, kernel_size, self._training)
        return y


class AvgPooling2d(Module):
    """AvgPooling layer used for downsampling.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the pooling window used for the pooling operation. Defaults to ``2``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.
    """

    def __init__(
        self, kernel_size: int = 2, label: Optional[str] = None, training: bool = False
    ) -> None:
        super().__init__(label, training)
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [4])

        kernel_size = (self.kernel_size, self.kernel_size)
        y, self._backward = avgpooling2d(x, kernel_size, self._training)
        return y
