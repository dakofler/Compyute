"""Neural network containers module"""

from walnut.tensor import Tensor, NpArrayLike
from walnut.nn.module import Module
from walnut.nn.layers import Linear, Recurrent


__all__ = ["SequentialContainer", "ParallelContainer", "RNN"]


class SequentialContainer(Module):
    """Sequential container."""

    def __init__(self, layers: list[Module]) -> None:
        """Sequential container module.

        Parameters
        ----------
        layers : list[Module]
            List of layers used in the container. These layers are processed sequentially.
        """
        super().__init__()
        self.sub_modules = layers

    def __call__(self, x: Tensor) -> Tensor:
        for module in self.sub_modules:
            x = module(x)

        if self.training:

            def backward(y_grad: NpArrayLike) -> NpArrayLike:
                self.set_y_grad(y_grad)

                for module in reversed(self.sub_modules):
                    y_grad = module.backward(y_grad)
                return y_grad

            self.backward = backward

        self.set_y(x)
        return x


class ParallelContainer(Module):
    """Parallel container."""

    def __init__(self, layers: list[Module]) -> None:
        super().__init__()
        self.sub_modules = layers

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError()


class RNN(SequentialContainer):
    """Recurrent neural network model."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        activation: str = "tanh",
        num_layers: int = 1,
        use_bias: bool = True,
    ) -> None:
        """Recurrent neural network model.

        Parameters
        ----------
        in_channels : int
            Number of input features.
        hidden_channels: int
            Number of hidden features.
        activation: Module, optional
            Activation function to be used in the hidden layer, by default Tanh().
        num_layers: int, optional
            Number of recurrent layers in the model, by default 1.
        use_bias : bool, optional
            Whether to use bias values, by default True.
        """
        modules = []
        for i in range(num_layers):
            if i == 0:
                modules.append(Linear(in_channels, hidden_channels, use_bias=use_bias))
            else:
                modules.append(
                    Linear(hidden_channels, hidden_channels, use_bias=use_bias)
                )
            modules.append(Recurrent(hidden_channels, activation, use_bias=use_bias))
        super().__init__(modules)
