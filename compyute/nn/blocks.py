"""Neural network blocks module"""

from compyute.nn.containers import SequentialContainer
from compyute.nn.layers import Linear, RecurrentCell
from compyute.nn.module import Module
from compyute.nn.parameter import Parameter
from compyute.tensor import Tensor, ArrayLike


__all__ = ["Recurrent", "Residual"]


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


class Residual(Module):
    """Residual connection block."""

    def __init__(self, layers: list[Module]) -> None:
        """Residual connection block."""
        super().__init__()
        self._block = SequentialContainer(layers)
        self.sub_modules = [self._block]

    def forward(self, x: Tensor) -> Tensor:
        y = x + self._block.forward(x)  # residual connection

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                dx = self._block.backward(dy) + dy

                return dx

            self.backward = backward

        self.set_y(y)
        return y
