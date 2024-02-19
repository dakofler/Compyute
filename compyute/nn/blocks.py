"""Neural network blocks module"""

from compyute.nn.layers import Linear, RecurrentCell
from compyute.nn.containers import SequentialContainer


__all__ = ["Recurrent"]


class Recurrent(SequentialContainer):
    """Recurrent neural network block."""

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        activation: str = "tanh",
        num_layers: int = 1,
        use_bias: bool = True,
    ) -> None:
        """Recurrent neural network block.

        Parameters
        ----------
        in_channels : int
            Number of input features.
        h_channels: int
            Number of hidden features used in each recurrent cell.
        activation: Module, optional
            Activation function to be used in hidden layers, by default "tanh".
        num_layers: int, optional
            Number of recurrent layers in the block, by default 1.
        use_bias : bool, optional
            Whether to use bias values, by default True.
        """
        modules = []
        for i in range(num_layers):
            if i == 0:
                modules.append(Linear(in_channels, h_channels, use_bias=use_bias))
            else:
                modules.append(Linear(h_channels, h_channels, use_bias=use_bias))
            modules.append(RecurrentCell(h_channels, activation, use_bias=use_bias))
        super().__init__(modules)
