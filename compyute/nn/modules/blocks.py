"""Neural network blocks module"""

from typing import Literal, Optional

from ...dtypes import Dtype, _DtypeLike
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
    """Dense neural network block containing a linear layer and an activation function."""

    __slots__ = ()

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
        """Dense neural network block.
        Input: (B, ... , Cin)
            B ... batch, Cin ... input channels
        Output: (B, ... , Co)
            B ... batch, Co ... output channels

        Parameters
        ----------
        in_channels : int
            Number of input features.
        out_channels : int
            Number of output channels (neurons) of the dense block.
        activation: _ActivationLike
            Activation function to use in the dense block.
        weight_init: _InitializerLike, optional
            What method to use for initializing weight parameters, by default "xavier_uniform".
        bias : bool, optional
            Whether to use bias values, by default True.
        bias_init: _InitializerLike, optional
            What method to use for initializing bias parameters, by default "zeros".
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default Dtype.FLOAT32.
        label: str, optional
            Module label.
        training: bool, optional
            Whether the module should be in training mode, by default False.
        """
        linear = Linear(in_channels, out_channels, bias, dtype, training=training)

        w_init = get_initializer(weight_init, dtype, activation)
        linear.w = Parameter(w_init((out_channels, in_channels)), "lin_w")

        if bias:
            b_init = get_initializer(bias_init, dtype, activation)
            linear.b = Parameter(b_init((out_channels,)), "lin_b")

        super().__init__(linear, get_activation(activation), label=label, training=training)


class Convolution1dBlock(Sequential):
    """Convolution 1d block containing a 1d convolutional layer and an activation function."""

    __slots__ = ()

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: _ActivationLike,
        kernel_size: int,
        padding: Literal["same", "valid"] = "valid",
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
        """Convolution 1d block containing a 1d convolutional layer and an activation function.
        Input: (B, Ci, Ti)
            B ... batch, Ci ... input channels, Ti ... input time
        Output: (B, Co, To)
            B ... batch, Co ... output channels, To ... output time

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels (filters).
        kernel_size : int
            Size of each kernel.
        activation: _ActivationLike
            Activation function to use in the dense block.
        padding: Literal["same", "valid"], optional
            Padding applied before convolution, by default "valid".
        stride : int, optional
            Stride used for the convolution operation, by default 1.
        dilation : int, optional
            Dilation used for each axis of the filter, by default 1.
        weight_init: _InitializerLike, optional
            What method to use for initializing weight parameters, by default "xavier_uniform".
        bias : bool, optional
            Whether to use bias values, by default True.
        bias_init: _InitializerLike, optional
            What method to use for initializing bias parameters, by default "zeros".
        batchnorm : bool, optional
            Whether to use batch normalization, by default False.
        batchnorm_eps : float, optional
            Constant for numerical stability used in batch normalization, by default 1e-5.
        batchnorm_m : float, optional
            Momentum used in batch normalization, by default 0.1.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default Dtype.FLOAT32.
        label: str, optional
            Module label.
        training: bool, optional
            Whether the module should be in training mode, by default False.
        """
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
        conv.w = Parameter(w_init((out_channels, in_channels, kernel_size)), "conv1d_w")

        if bias:
            b_init = get_initializer(bias_init, dtype, activation)
            conv.b = Parameter(b_init((out_channels,)), "conv1d_b")

        if batchnorm:
            bn = Batchnorm1d(out_channels, batchnorm_eps, batchnorm_m, dtype, training=training)
            super().__init__(conv, bn, get_activation(activation), label=label, training=training)
        else:
            super().__init__(conv, get_activation(activation), label=label, training=training)


class Convolution2dBlock(Sequential):
    """Convolution 2d block containing a 2d convolutional layer and an activation function."""

    __slots__ = ()

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: _ActivationLike,
        kernel_size: int = 3,
        padding: Literal["same", "valid"] = "valid",
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
        """Convolution 2d block containing a 2d convolutional layer and an activation function.
        Input: (B, Ci, Yi, Xi)
            B ... batch, Ci ... input channels, Yi ... input height, Xi ... input width
        Output: (B, Co, Yo, Xo)
            B ... batch, Co ... output channels, Yo ... output height, Xo ... output width

        Parameters
        ----------
        in_channels : int
            Number of input channels (color channels).
        out_channels : int
            Number of output channels (filters).
        kernel_size : int, optional
            Size of each kernel, by default 3.
        activation: _ActivationLike
            Activation function to use in the dense block.
        padding: Literal["same", "valid"], optional
            Padding applied before convolution, by default "valid".
        stride : int , optional
            Strides used for the convolution operation, by default 1.
        dilation : int , optional
            Dilations used for each axis of the filter, by default 1.
        weight_init: _InitializerLike, optional
            What method to use for initializing weight parameters, by default "xavier_uniform".
        bias : bool, optional
            Whether to use bias values, by default True.
        bias_init: _InitializerLike, optional
            What method to use for initializing bias parameters, by default "zeros".
        batchnorm : bool, optional
            Whether to use batch normalization, by default False.
        batchnorm_eps : float, optional
            Constant for numerical stability used in batch normalization, by default 1e-5.
        batchnorm_m : float, optional
            Momentum used in batch normalization, by default 0.1.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default Dtype.FLOAT32.
        label: str, optional
            Module label.
        training: bool, optional
            Whether the module should be in training mode, by default False.
        """
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
        conv.w = Parameter(
            w_init((out_channels, in_channels, kernel_size, kernel_size)), "conv2d_w"
        )

        if bias:
            b_init = get_initializer(bias_init, dtype, activation)
            conv.b = Parameter(b_init((out_channels,)), "conv2d_b")

        if batchnorm:
            bn = Batchnorm2d(out_channels, batchnorm_eps, batchnorm_m, dtype, training=training)
            super().__init__(conv, bn, get_activation(activation), label=label, training=training)
        else:
            super().__init__(conv, get_activation(activation), label=label, training=training)


class ResidualBlock(ParallelAdd):
    """Residual block implementing a residual connection bypassing a block of modules."""

    __slots__ = ()

    def __init__(
        self,
        *modules: Module,
        residual_proj: Optional[Module] = None,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        """Residual block implementing a residual connection bypassing a block of modules.

        Parameters
        ----------
        *modules : Module
            Modules used in the residual block.
        residual_projection: Module, optional
            Module used for a linear projection to achieve matching dimensions, by default None.
            If none is provided, the identity function is used.
        label: str, optional
            Module label.
        training: bool, optional
            Whether the module should be in training mode, by default False.
        """
        proj = residual_proj if residual_proj is not None else Identity(training=training)

        if len(modules) == 1:
            super().__init__(modules[0], proj, label=label, training=training)
        else:
            module_block = Sequential(*modules, label=label, training=training)
            super().__init__(module_block, proj, label=label, training=training)
