"""Neural network convolution modules."""

from typing import Literal, Optional

from ...tensor_ops.creating import empty
from ...tensors import Tensor
from ...typing import DType
from ..functional.convolution_funcs import (
    AvgPooling2DFn,
    Convolution1DFn,
    Convolution2DFn,
    MaxPooling2DFn,
)
from ..parameter import Parameter, update_parameter_grad
from ..utils.initializers import init_xavier_uniform, init_zeros
from .module import Module, validate_input_axes

__all__ = ["Convolution1D", "Convolution2D", "MaxPooling2D", "AvgPooling2D"]


PaddingLike = int | Literal["valid", "same"]


def _str_to_pad(
    padding: Literal["valid", "same"], kernel_size: int, dilation: int
) -> int:
    if padding == "valid":
        return 0
    return (kernel_size * dilation - 1) // 2


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
        self.padding = (
            padding
            if isinstance(padding, int)
            else _str_to_pad(padding, kernel_size, dilation)
        )
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        # init parameters
        self.w = Parameter(empty((out_channels, in_channels, kernel_size), dtype=dtype))
        self.b = Parameter(empty((out_channels,), dtype=dtype)) if bias else None
        self._init_parameters_and_buffers()

    def _init_parameters_and_buffers(self) -> None:
        init_xavier_uniform(self.w)
        if self.b:
            init_zeros(self.b)

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [3])
        return Convolution1DFn.forward(
            self.fcache, x, self.w, self.b, self.padding, self.stride, self.dilation
        )

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw, db = Convolution1DFn.backward(self.fcache, dy)
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
        self.padding = (
            padding
            if isinstance(padding, int)
            else _str_to_pad(padding, kernel_size, dilation)
        )
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
        init_xavier_uniform(self.w)
        if self.b:
            init_zeros(self.b)

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [4])
        return Convolution2DFn.forward(
            self.fcache, x, self.w, self.b, self.padding, self.stride, self.dilation
        )

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw, db = Convolution2DFn.backward(self.fcache, dy)
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

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [4])
        return MaxPooling2DFn.forward(self.fcache, x, self.kernel_size)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return MaxPooling2DFn.backward(self.fcache, dy)


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

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [4])
        return AvgPooling2DFn.forward(self.fcache, x, self.kernel_size)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return AvgPooling2DFn.backward(self.fcache, dy)
