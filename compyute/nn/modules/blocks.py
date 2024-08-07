"""Neural network block modules."""

from typing import Optional

from ...dtypes import Dtype, _DtypeLike
from ..functional.convolutions import _PaddingLike
from ..initializers import _InitializerLike, get_initializer
from ..parameter import Parameter
from .activations import _ActivationLike, get_activation
from .containers import ParallelAdd, Sequential
from .convolution import Convolution1d, Convolution2d
from .linear import Linear
from .module import Identity, Module
from .normalization import Batchnorm1d, Batchnorm2d

__all__ = ["Convolution1dBlock", "Convolution2dBlock", "DenseBlock", "ResidualBlock"]


class DenseBlock(Sequential):
    """Dense neural network block containing a linear transformation layer
    and an activation function.

    Shapes:
        - Input :math:`(B_1, ... , B_n, C_{in})`
        - Output :math:`(B_1, ... , B_n, C_{out})`
    where
        - :math:`B_1, ... , B_n` ... batch axes
        - :math:`C_{in}` ... input channels
        - :math:`C_{out}` ... output channels

    Parameters
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output channels (neurons) of the dense block.
    activation : _ActivationLike
        Activation function to use in the dense block.
        See :ref:`activations` for more details.
    weight_init : _InitializerLike, optional
        What method to use for initializing weight parameters. Defaults to ``xavier_uniform``.
        See :ref:`initializers` for more details.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    bias_init : _InitializerLike, optional
        What method to use for initializing bias parameters. Defaults to ``zeros``.
        See :ref:`initializers` for more details.
    dtype : DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.


    See Also
    --------
    :class:`compyute.nn.Linear`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: _ActivationLike,
        weight_init: _InitializerLike = "xavier_uniform",
        bias: bool = True,
        bias_init: _InitializerLike = "zeros",
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        linear = Linear(in_channels, out_channels, bias, dtype, training=training)

        w_init = get_initializer(weight_init, dtype, activation)
        linear.w = Parameter(w_init((out_channels, in_channels)))

        if bias:
            b_init = get_initializer(bias_init, dtype, activation)
            linear.b = Parameter(b_init((out_channels,)))

        super().__init__(linear, get_activation(activation), label=label, training=training)


class Convolution1dBlock(Sequential):
    """Convolution block containing a 1D convolutional layer, followed by an
    optional batch normalization and an activation function.

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
    activation : _ActivationLike
        Activation function to use in the dense block.
        See :ref:`activations` for more details.
    padding : _PaddingLike, optional
        Padding applied before convolution. Defaults to ``valid``.
    stride : int, optional
        Stride used for the convolution operation. Defaults to ``1``.
    dilation : int, optional
        Dilation used for each axis of the filter. Defaults to ``1``.
    weight_init : _InitializerLike, optional
        What method to use for initializing weight parameters. Defaults to ``xavier_uniform``.
        See :ref:`initializers` for more details.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    bias_init : _InitializerLike, optional
        What method to use for initializing bias parameters. Defaults to ``zeros``.
        See :ref:`initializers` for more details.
    batchnorm : bool, optional
        Whether to use batch normalization. Defaults to ``False``.
    batchnorm_eps : float, optional
        Constant for numerical stability used in batch normalization. Defaults to ``1e-5``.
    batchnorm_m : float, optional
        Momentum used in batch normalization. Defaults to ``0.1``.
    dtype : DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.


    See Also
    --------
    :class:`compyute.nn.Convolution1d`
    :class:`compyute.nn.Batchnorm1d`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: _ActivationLike,
        padding: _PaddingLike = "valid",
        stride: int = 1,
        dilation: int = 1,
        weight_init: _InitializerLike = "xavier_uniform",
        bias: bool = True,
        bias_init: _InitializerLike = "zeros",
        batchnorm: bool = False,
        batchnorm_eps: float = 1e-5,
        batchnorm_m: float = 0.1,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        conv = Convolution1d(
            in_channels,
            out_channels,
            kernel_size,
            padding,
            stride,
            dilation,
            bias,
            dtype,
            training=training,
        )

        w_init = get_initializer(weight_init, dtype, activation)
        conv.w = Parameter(w_init((out_channels, in_channels, kernel_size)))

        if bias:
            b_init = get_initializer(bias_init, dtype, activation)
            conv.b = Parameter(b_init((out_channels,)))

        if batchnorm:
            bn = Batchnorm1d(out_channels, batchnorm_eps, batchnorm_m, dtype, training=training)
            super().__init__(conv, bn, get_activation(activation), label=label, training=training)
        else:
            super().__init__(conv, get_activation(activation), label=label, training=training)


class Convolution2dBlock(Sequential):
    """Convolution block containing a 2D convolutional layer, followed by an
    optional batch normalization and an activation function.

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
    activation : _ActivationLike
        Activation function to use in the dense block.
        See :ref:`activations` for more details.
    padding : _PaddingLike, optional
        Padding applied before convolution. Defaults to ``valid``.
    stride : int , optional
        Strides used for the convolution operation. Defaults to ``1``.
    dilation : int , optional
        Dilations used for each axis of the filter. Defaults to ``1``.
    weight_init : _InitializerLike, optional
        What method to use for initializing weight parameters. Defaults to ``xavier_uniform``.
        See :ref:`initializers` for more details.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    bias_init : _InitializerLike, optional
        What method to use for initializing bias parameters. Defaults to ``zeros``.
        See :ref:`initializers` for more details.
    batchnorm : bool, optional
        Whether to use batch normalization. Defaults to ``False``.
    batchnorm_eps : float, optional
        Constant for numerical stability used in batch normalization. Defaults to ``1e-5``.
    batchnorm_m : float, optional
        Momentum used in batch normalization. Defaults to ``0.1``.
    dtype : DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.


    See Also
    --------
    :class:`compyute.nn.Convolution2d`
    :class:`compyute.nn.Batchnorm2d`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: _ActivationLike,
        padding: _PaddingLike = "valid",
        stride: int = 1,
        dilation: int = 1,
        weight_init: _InitializerLike = "xavier_uniform",
        bias: bool = True,
        bias_init: _InitializerLike = "zeros",
        batchnorm: bool = False,
        batchnorm_eps: float = 1e-5,
        batchnorm_m: float = 0.1,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        conv = Convolution2d(
            in_channels,
            out_channels,
            kernel_size,
            padding,
            stride,
            dilation,
            bias,
            dtype,
            training=training,
        )

        w_init = get_initializer(weight_init, dtype, activation)
        conv.w = Parameter(w_init((out_channels, in_channels, kernel_size, kernel_size)))

        if bias:
            b_init = get_initializer(bias_init, dtype, activation)
            conv.b = Parameter(b_init((out_channels,)))

        if batchnorm:
            bn = Batchnorm2d(out_channels, batchnorm_eps, batchnorm_m, dtype, training=training)
            super().__init__(conv, bn, get_activation(activation), label=label, training=training)
        else:
            super().__init__(conv, get_activation(activation), label=label, training=training)


class ResidualBlock(ParallelAdd):
    """Residual block implementing a residual connection around a block of modules.
    Modules in the residual block are processed sequentially.

    Parameters
    ----------
    *modules : Module
        Modules used in the residual block. They are processed sequentially.
    residual_projection : Module, optional
        Module used as a projection to achieve matching dimensions. Defaults to ``None``.
        If ``None``, the identity function is used.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the module should be in training mode. Defaults to ``False``.
    """

    def __init__(
        self,
        *modules: Module,
        residual_proj: Optional[Module] = None,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        proj = residual_proj if residual_proj is not None else Identity(training=training)

        if len(modules) == 1:
            super().__init__(modules[0], proj, label=label, training=training)
        else:
            module_block = Sequential(*modules, label=label, training=training)
            super().__init__(module_block, proj, label=label, training=training)
