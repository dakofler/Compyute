"""recurrent layers layer"""


import numpy as np

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, NumpyArray
from walnut.nn.module import Module
from walnut.nn.funcional import relu
from walnut.nn.layers import Linear


__all__ = ["Recurrent", "RNN"]


class Recurrent(Module):
    """Recurrent layer."""

    def __init__(
        self,
        hidden_channels: int,
        activation: str = "tanh",
        weights: Tensor | None = None,
        use_bias: bool = True,
    ) -> None:
        """Recurrent layer.

        Parameters
        ----------
        hidden_channels : int
            Number of hidden channels of the layer.
        activation : str, optional
            Activation function used in the recurrent layer, by default "tanh".
        weights : Tensor | None, optional
            Weights of the layer, by default None. If None, weights are initialized randomly.
        use_bias : bool, optional
            Whether to use bias values, by default True.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.activation = activation
        self.use_bias = use_bias

        # init weights (c_hidden, c_hidden)
        if weights is None:
            k = hidden_channels**-0.5
            self.w = tu.randu((hidden_channels, hidden_channels), -k, k)
        else:
            self.w = weights
        self.parameters.append(self.w)

        # init bias (c_out,)
        if use_bias:
            self.b = tu.zeros((hidden_channels,))
            self.parameters.append(self.b)

    def __repr__(self):
        name = self.__class__.__name__
        hidden_channels = self.hidden_channels
        activation = self.activation
        use_bias = self.use_bias
        return f"{name} ({hidden_channels=}, {activation=}, {use_bias=})"

    def __call__(self, x: Tensor) -> Tensor:
        y = tu.zeros_like(x)  # (B, T, H)
        h = tu.zeros_like(x)  # (B, T, H)

        for i in range(x.shape[1]):
            # hidden states
            h[:, i] = y[:, i - 1] @ self.w  # (B, t, H)
            if self.use_bias:
                h[:, i] += self.b

            # activation
            if self.activation == "tanh":
                y[:, i] = (x[:, i] + h[:, i]).tanh()
            else:
                y[:, i] = relu(x[:, i] + h[:, i])

        if self.training:

            def backward(y_grad: NumpyArray) -> NumpyArray:
                self.set_y_grad(y_grad)
                x_grad = np.zeros_like(x.data)

                for i in range(x.shape[1] - 1, -1, -1):
                    # add hidden state gradient of next layer, if not last sequence element
                    if i == x.shape[1] - 1:
                        grad = y_grad[:, i]
                    else:
                        grad = y_grad[:, i] + x_grad[:, i + 1] @ self.w.T

                    # activation gradient
                    if self.activation == "tanh":
                        act_grad = -y.data[:, i] ** 2 + 1.0
                    else:
                        act_grad = y.data[:, i] > 0
                    x_grad[:, i] = act_grad * grad

                    # weight grads
                    if i > 0:
                        self.w.grad = y[:, i - 1].T @ x_grad[:, i]

                # bias grads
                self.b.grad = x_grad.sum((0, 1))

                return x_grad

            self.backward = backward

        self.set_y(y)
        return y


class RNN(Module):
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
        super().__init__()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(
                    Linear(in_channels, hidden_channels, use_bias=use_bias)
                )
            else:
                self.layers.append(
                    Linear(hidden_channels, hidden_channels, use_bias=use_bias)
                )
            self.layers.append(
                Recurrent(hidden_channels, activation, use_bias=use_bias)
            )

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        if self.training:

            def backward(y_grad: NumpyArray) -> NumpyArray:
                layers_reversed = self.layers.copy()
                layers_reversed.reverse()

                for layer in layers_reversed:
                    y_grad = layer.backward(y_grad)
                return y_grad

            self.backward = backward

        return x
