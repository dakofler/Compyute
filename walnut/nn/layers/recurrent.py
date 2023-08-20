"""recurrent layers layer"""


import numpy as np

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, NumpyArray
from walnut.nn.module import Module
from walnut.nn.layers import Linear


__all__ = ["Recurrent"]


class Recurrent(Module):
    """Recurrent layer."""

    def __init__(self, hidden_channels: int, activation: Module) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden = Linear(hidden_channels, hidden_channels)
        self.activation = activation
        self.layers = [self.hidden, self.activation]

    def __repr__(self):
        name = self.__class__.__name__
        hidden_channels = self.hidden_channels
        activation = self.activation.__repr__()
        return f"{name} ({hidden_channels=}, {activation=})"

    def __call__(self, x: Tensor) -> Tensor:
        y = tu.zeros_like(x)  # (B, T, H)
        h = tu.zeros_like(x)  # (B, T, H)

        for i in range(x.shape[1]):
            h[:, i] = self.hidden(y[:, i - 1])  # (B, t, H) @ (H, H)
            y[:, i] = self.activation(x[:, i] + h[:, i])  # (B, t, H)

        if self.training:

            def backward(y_grad: NumpyArray) -> NumpyArray:
                self.set_y_grad(y_grad)

                x_grad = np.zeros_like(x.data)
                h_grad = np.zeros_like(h.data)

                # Problem: cannot backward multiple times or different inputs,
                # because backward is defined during call and therefore data
                # of the last forward pass is used for x, y, etc.

                for i in range(x.shape[1] - 1, -1, -1):
                    if i == x.shape[1] - 1:
                        x_grad[:, i] = self.activation.backward(y_grad[:, i])
                    else:
                        grad = y_grad[:, i] + h_grad[:, i + 1]
                        x_grad[:, i] = self.activation.backward(grad)
                        h_grad[:, i] = self.hidden.backward(x_grad[:, i])

                return x_grad

            self.backward = backward

        self.set_y(y)
        return y
