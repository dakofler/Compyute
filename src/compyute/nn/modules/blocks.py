"""Neural network blocks module"""

from typing import Literal, Optional

from ...types import _DtypeLike
from .activations import get_act_from_str
from .containers import ParallelAdd, Sequential
from .convolution import Convolution1d, Convolution2d
from .linear import Linear
from .module import Module

__all__ = ["Convolution1dBlock", "Convolution2dBlock", "DenseBlock", "SkipConnection"]


class DenseBlock(Sequential):
    """Dense neural network block containing a linear layer and an activation function."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Literal["relu", "leaky_relu", "gelu", "sigmoid", "tanh"],
        bias: bool = True,
        dtype: _DtypeLike = "float32",
        label: Optional[str] = None,
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
        activation: Literal["relu", "leaky_relu", "gelu", "sigmoid", "tanh"]
            Activation function to use in the dense block.
        bias : bool, optional
            Whether to use bias values, by default True.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        label: str, optional
            Module label.
        """
        super().__init__(
            Linear(in_channels, out_channels, bias, dtype),
            get_act_from_str(activation),
            label=label,
        )


class Convolution1dBlock(Sequential):
    """Convolution 1d block containing a 1d convolutional layer and an activation function."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Literal["relu", "leaky_relu", "gelu", "sigmoid", "tanh"],
        kernel_size: int,
        padding: Literal["causal", "same", "valid"] = "causal",
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        dtype: _DtypeLike = "float32",
        label: Optional[str] = None,
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
        activation: Literal["relu", "leaky_relu", "gelu", "sigmoid", "tanh"]
            Activation function to use in the dense block.
        padding: Literal["causal", "same", "valid"], optional
            Padding applied before convolution, by default "causal".
        stride : int, optional
            Stride used for the convolution operation, by default 1.
        dilation : int, optional
            Dilation used for each axis of the filter, by default 1.
        bias : bool, optional
            Whether to use bias values, by default True.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        label: str, optional
            Module label.
        """
        super().__init__(
            Convolution1d(
                in_channels,
                out_channels,
                kernel_size,
                padding,
                stride,
                dilation,
                bias,
                dtype,
            ),
            get_act_from_str(activation),
            label=label,
        )


class Convolution2dBlock(Sequential):
    """Convolution 2d block containing a 2d convolutional layer and an activation function."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Literal["relu", "leaky_relu", "gelu", "sigmoid", "tanh"],
        kernel_size: int = 3,
        padding: Literal["same", "valid"] = "valid",
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        dtype: _DtypeLike = "float32",
        label: Optional[str] = None,
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
        activation: Literal["relu", "leaky_relu", "gelu", "sigmoid", "tanh"]
            Activation function to use in the dense block.
        padding: Literal["same", "valid"], optional
            Padding applied before convolution, by default "valid".
        stride : int , optional
            Strides used for the convolution operation, by default 1.
        dilation : int , optional
            Dilations used for each axis of the filter, by default 1.
        bias : bool, optional
            Whether to use bias values, by default True.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        label: str, optional
            Module label.
        """
        super().__init__(
            Convolution2d(
                in_channels,
                out_channels,
                kernel_size,
                padding,
                stride,
                dilation,
                bias,
                dtype,
            ),
            get_act_from_str(activation),
            label=label,
        )


class SkipConnection(ParallelAdd):
    """Skip connection bypassing a block of modules."""

    def __init__(self, residual_block: Module, projection: Optional[Module] = None) -> None:
        """Residual connection bypassing a block of modules.

        Parameters
        ----------
        residual_block : Module
            Residual block bypassed by the skip connection.
            For multiple modules use a container module.
        projection: Module, optional
            Module used for a linear projection to achieve matching dimensions, by default None.
        label: str, optional
            Module label.
        """
        proj = projection if projection is not None else Module("Projection")
        super().__init__(residual_block, proj)
