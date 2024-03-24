"""Neural network blocks module"""

from .containers import SequentialContainer, ParallelAddContainer
from .layers import RecurrentCell
from .module import Module


__all__ = ["RecurrentBlock", "ResidualBlock"]


class RecurrentBlock(SequentialContainer):
    """Recurrent neural network block."""

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        num_layers: int = 1,
        use_bias: bool = True,
        dtype: str = "float32",
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
