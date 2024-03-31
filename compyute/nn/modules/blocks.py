"""Neural network blocks module"""

from typing import Literal
from .containers import SequentialContainer, ParallelAddContainer
from .layers import Linear
from .layers.activations import get_act_from_str
from .module import Module
from ...types import DtypeLike


__all__ = ["DenseBlock", "ResidualBlock"]


class DenseBlock(SequentialContainer):
    """Dense neural network block."""

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
