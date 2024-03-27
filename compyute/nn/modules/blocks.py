"""Neural network blocks module"""

from typing import Literal
from .containers import SequentialContainer, ParallelAddContainer
from .layers import RecurrentCell, Linear
from .layers.activations import get_act_from_str
from .module import Module
from ...types import DtypeLike


__all__ = ["DenseBlock", "RecurrentBlock", "ResidualBlock"]


class DenseBlock(SequentialContainer):
    """Dense neural network block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Literal["relu", "leaky_relu", "gelu", "sigmoid", "tanh"],
        use_bias: bool = True,
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
        use_bias : bool, optional
            Whether to use bias values, by default True.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        """
        layers = [
            Linear(in_channels, out_channels, use_bias, dtype),
            get_act_from_str(activation),
        ]
        super().__init__(layers)


class RecurrentBlock(SequentialContainer):
    """Recurrent neural network block."""

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        num_layers: int = 1,
        use_bias: bool = True,
        dtype: DtypeLike = "float32",
    ) -> None:
        """Recurrent neural network block.
        Input: (B, T , Cin)
            B ... batch, T ... time, Cin ... input channels
        Output: (B, T, Ch)
            B ... batch, T ... time, Ch ... hidden channels

        Parameters
        ----------
        in_channels : int
            Number of input features.
        h_channels: int
            Number of hidden features used in each recurrent cell.
        num_layers: int, optional
            Number of recurrent layers in the block, by default 1.
        use_bias : bool, optional
            Whether to use bias values, by default True.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default "float32".
        """
        m = [RecurrentCell(in_channels, h_channels, use_bias=use_bias, dtype=dtype)]
        for _ in range(num_layers - 1):
            m.append(
                RecurrentCell(h_channels, h_channels, use_bias=use_bias, dtype=dtype)
            )
        super().__init__(m)


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
