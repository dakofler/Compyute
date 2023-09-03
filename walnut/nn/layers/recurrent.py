"""recurrent layers layer"""

import numpy as np

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, NpArrayLike
from walnut.nn.module import Module
from walnut.nn.funcional import relu


__all__ = ["Recurrent"]


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
        self.parameters = [self.w]

        # init bias (c_out,)
        if use_bias:
            self.b = tu.zeros((hidden_channels,))
            self.parameters += [self.b]

    def __repr__(self):
        name = self.__class__.__name__
        hidden_channels = self.hidden_channels
        activation = self.activation
        use_bias = self.use_bias
        return f"{name}({hidden_channels=}, {activation=}, {use_bias=})"

    def __call__(self, x: Tensor) -> Tensor:
        y = tu.zeros_like(x)

        # iterate over sequence elements
        for i in range(x.shape[1]):
            # hidden states
            h = y[:, i - 1] @ self.w
            if self.use_bias:
                h += self.b

            # activation
            if self.activation == "tanh":
                y[:, i] = (x[:, i] + h).tanh()
            else:
                y[:, i] = relu(x[:, i] + h)

        if self.training:

            def backward(y_grad: NpArrayLike) -> NpArrayLike:
                self.set_y_grad(y_grad)
                x_grad = np.zeros_like(x.data)
                self.w.grad = np.zeros_like(self.w.data)

                for i in range(x.shape[1] - 1, -1, -1):
                    # add hidden state gradient of next layer, if not last sequence element
                    if i == x.shape[1] - 1:
                        out_grad = y_grad[:, i]
                    else:
                        out_grad = y_grad[:, i] + x_grad[:, i + 1] @ self.w.T

                    # activation gradient
                    if self.activation == "tanh":
                        act_grad = -y.data[:, i] ** 2 + 1.0
                    else:
                        act_grad = y.data[:, i] > 0
                    x_grad[:, i] = act_grad * out_grad

                    # weight grads
                    if i > 0:
                        self.w.grad += y[:, i - 1].T @ x_grad[:, i]

                # bias grads
                self.b.grad = x_grad.sum((0, 1))

                return x_grad

            self.backward = backward

        self.set_y(y)
        return y
