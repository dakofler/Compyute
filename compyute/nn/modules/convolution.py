"""Neural network convolution modules."""

from typing import Literal, Optional

from ...tensor_ops.creating import empty
from ...tensors import Tensor
from ...typing import DType
from ..functional.convolutions import (
    FAvgPooling2D,
    FConvolution1D,
    FConvolution2D,
    FMaxPooling2D,
    PaddingLike,
)
from ..parameter import Parameter, update_parameter_grad
from ..utils.initializers import XavierUniform, Zeros
from .module import Module, validate_input_axes

__all__ = ["Convolution1D", "Convolution2D", "MaxPooling2D", "AvgPooling2D"]


class Convolution1D(Module):
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
    padding : PaddingLike, optional
        Padding applied before convolution. Defaults to ``valid``.
    stride : int, optional
        Stride used for the convolution operation. Defaults to ``1``.
    dilation : int, optional
        Dilation used for each axis of the filter. Defaults to ``1``.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    dtype : DType, optional
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in} \cdot \text{kernel_size}}}`. Biases are initialized as zeros.
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
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        # init parameters
        self.w = Parameter(empty((out_channels, in_channels, kernel_size), dtype=dtype))
        self.b = Parameter(empty((out_channels,), dtype=dtype)) if bias else None
        self._init_parameters_and_buffers()

    def _init_parameters_and_buffers(self) -> None:
        XavierUniform()(self.w)
        if self.b:
            Zeros()(self.b)

    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [3])
        return FConvolution1D.forward(
            self._fcache, x, self.w, self.b, self.padding, self.stride, self.dilation
        )

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        dx, dw, db = FConvolution1D.backward(
            self._fcache, dy, self.padding, self.stride, self.dilation
        )
        update_parameter_grad(self.w, dw)
        update_parameter_grad(self.b, db)
        return dx


class Convolution2D(Module):
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
    padding : PaddingLike, optional
        Padding applied before convolution. Defaults to ``valid``.
    stride : int , optional
        Strides used for the convolution operation. Defaults to ``1``.
    dilation : int , optional
        Dilations used for each axis of the filter. Defaults to ``1``.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.


    .. note::
        Weights are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in} * \text{kernel_size}^2}}`. Biases are initialized as zeros.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: PaddingLike = "valid",
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        # init parameters
        self.w = Parameter(
            empty((out_channels, in_channels, kernel_size, kernel_size), dtype=dtype)
        )
        self.b = Parameter(empty((out_channels,), dtype=dtype)) if bias else None
        self._init_parameters_and_buffers()

    def _init_parameters_and_buffers(self) -> None:
        XavierUniform()(self.w)
        if self.b:
            Zeros()(self.b)

    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [4])
        return FConvolution2D.forward(
            self._fcache, x, self.w, self.b, self.padding, self.stride, self.dilation
        )

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        dx, dw, db = FConvolution2D.backward(
            self._fcache, dy, self.padding, self.stride, self.dilation
        )
        update_parameter_grad(self.w, dw)
        update_parameter_grad(self.b, db)
        return dx


class MaxPooling2D(Module):
    """Pooling layer used for downsampling where the
    maximum value within the pooling window is used.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the pooling window used for the pooling operation. Defaults to ``2``.
    """

    def __init__(self, kernel_size: int = 2, label: Optional[str] = None) -> None:
        super().__init__(label)
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [4])
        kernel_size = (self.kernel_size, self.kernel_size)
        return FMaxPooling2D.forward(self._fcache, x, kernel_size)

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        kernel_size = (self.kernel_size, self.kernel_size)
        return FMaxPooling2D.backward(self._fcache, dy, kernel_size)


class AvgPooling2D(Module):
    """Pooling layer used for downsampling where the
    average value within the pooling window is used.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the pooling window used for the pooling operation. Defaults to ``2``.
    """

    def __init__(self, kernel_size: int = 2, label: Optional[str] = None) -> None:
        super().__init__(label)
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [4])
        kernel_size = (self.kernel_size, self.kernel_size)
        return FAvgPooling2D.forward(self._fcache, x, kernel_size)

    def backward(self, dy: Tensor) -> Tensor:
        super().backward(dy)
        kernel_size = (self.kernel_size, self.kernel_size)
        return FAvgPooling2D.backward(self._fcache, dy, kernel_size)
