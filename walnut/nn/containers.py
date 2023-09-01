"""Neural network containers module"""

from walnut.tensor import Tensor, NpArrayLike
from walnut.nn.module import Module
from walnut.nn.layers import Linear, Recurrent


__all__ = ["SequentialContainer", "ParallelContainer", "RNN"]


class Container(Module):
    """Container module."""

    def __init__(self) -> None:
        """Container.

        Parameters
        ----------
        layers : list[Module]
            List of layers.
        """
        super().__init__()
        self.layers: list[Module] = []

    def __repr__(self) -> str:
        string = super().__repr__()

        if len(self.layers) > 0:
            for layer in self.layers:
                string += "\n" + layer.__repr__()

        return string

    def get_parameters(self):
        """Returns parameters of the cointainer's layers."""
        parameters = super().get_parameters()
        for layer in self.layers:
            parameters += layer.get_parameters()
        return parameters

    def reset_grads(self):
        """Resets parameter grads to improve memory usage."""
        super().reset_grads()
        for layer in self.layers:
            layer.reset_grads()

    def training_mode(self):
        """Puts the module into training mode. Some modules may have different forward
        behaviour if in training mode. Backward behaviour is only defined in training mode.
        """
        super().training_mode()
        for layer in self.layers:
            layer.training_mode()

    def eval_mode(self):
        """Puts the module into evaluation mode. Some modules may have different forward
        behaviour if in training mode. Backward behaviour is only defined in training mode.
        """
        super().eval_mode()
        for layer in self.layers:
            layer.eval_mode()


class SequentialContainer(Container):
    """Sequential container."""

    def __init__(self, layers: list[Module]) -> None:
        super().__init__()
        self.layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        if self.training:

            def backward(y_grad: NpArrayLike) -> NpArrayLike:
                self.set_y_grad(y_grad)

                for layer in reversed(self.layers):
                    y_grad = layer.backward(y_grad)
                return y_grad

            self.backward = backward

        self.set_y(x)
        return x


class ParallelContainer(Container):
    """Parallel container."""

    def __init__(self, layers: list[Module]) -> None:
        super().__init__()
        self.layers = layers

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
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(Linear(in_channels, hidden_channels, use_bias=use_bias))
            else:
                layers.append(
                    Linear(hidden_channels, hidden_channels, use_bias=use_bias)
                )
            layers.append(Recurrent(hidden_channels, activation, use_bias=use_bias))
        super().__init__(layers)
