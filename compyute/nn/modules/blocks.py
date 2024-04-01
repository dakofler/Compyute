"""Neural network blocks module"""

from typing import Literal
from .containers import SequentialContainer, ParallelAddContainer
from .layers import Convolution1d, Convolution2d, Linear
from .layers.activations import get_act_from_str
from .module import Module
from ...types import DtypeLike


__all__ = ["Convolution1dBlock", "Convolution2dBlock", "DenseBlock", "ResidualBlock"]


class DenseBlock(SequentialContainer):
    """Dense neural network block containing a linear layer and an activation function."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Literal["relu", "leaky_relu", "gelu", "sigmoid", "tanh"],
        bias: bool = True,
        dtype: DtypeLike = "float32",
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
        """
        layers = [
            Linear(in_channels, out_channels, bias, dtype),
            get_act_from_str(activation),
        ]
        super().__init__(layers)


class Convolution1dBlock(SequentialContainer):
    """Convolution 1d block containing a 1d convolutional layer and an activation function."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Literal["relu", "leaky_relu", "gelu", "sigmoid", "tanh"],
        kernel_size: int,
        pad: str = "causal",
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        dtype: DtypeLike = "float32",
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
        pad: str, optional
            Padding applied before convolution.
            Options are "causal", "valid" or "same", by default "causal".
        stride : int, optional
            Stride used for the convolution operation, by default 1.
        dilation : int, optional
            Dilation used for each axis of the filter, by default 1.
        bias : bool, optional
            Whether to use bias values, by default True.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        """
        layers = [
            Convolution1d(
                in_channels,
                out_channels,
                kernel_size,
                pad,
                stride,
                dilation,
                bias,
                dtype,
            ),
            get_act_from_str(activation),
        ]
        super().__init__(layers)


class Convolution2dBlock(SequentialContainer):
    """Convolution 2d block containing a 2d convolutional layer and an activation function."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Literal["relu", "leaky_relu", "gelu", "sigmoid", "tanh"],
        kernel_size: int = 3,
        pad: str = "valid",
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        dtype: DtypeLike = "float32",
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
        pad: str, optional
            Padding applied before convolution.
            Options are "valid" and "same", by default "valid".
        stride : int , optional
            Strides used for the convolution operation, by default 1.
        dilation : int , optional
            Dilations used for each axis of the filter, by default 1.
        bias : bool, optional
            Whether to use bias values, by default True.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        """
        layers = [
            Convolution2d(
                in_channels,
                out_channels,
                kernel_size,
                pad,
                stride,
                dilation,
                bias,
                dtype,
            ),
            get_act_from_str(activation),
        ]
        super().__init__(layers)


class ResidualBlock(ParallelAddContainer):
    """Block with residual connection."""

    def __init__(self, core_module: Module) -> None:
        """Block with residual connection bypassing the core module.

        Parameters
        ----------
        core_module : Module
            Core module bypassed by the residual connection.
            For multiple modules use a container as core module.
            To ensure matching tensor shapes, you might need to use a projection layer.
        """
        super().__init__([core_module, Module()])  # emtpy module as residual connection
