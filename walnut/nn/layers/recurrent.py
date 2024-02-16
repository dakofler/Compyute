"""recurrent layers layer"""

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ArrayLike
from walnut.nn.funcional import relu
from walnut.nn.module import Module
from walnut.nn.parameter import Parameter


__all__ = ["RecurrentCell"]


class RecurrentCell(Module):
    """Recurrent layer."""

    def __init__(
        self,
        hidden_channels: int,
        activation: str = "tanh",
        weights: Parameter | None = None,
        use_bias: bool = True,
        dtype: str = "float32",
    ) -> None:
        """Recurrent layer.

        Parameters
        ----------
        hidden_channels : int
            Number of hidden channels of the layer.
        activation : str, optional
            Activation function used in the recurrent layer, by default "tanh".
        weights : Parameter | None, optional
            Weights of the layer, by default None. If None, weights are initialized randomly.
        use_bias : bool, optional
            Whether to use bias values, by default True.
        dtype: str, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.activation = activation
        self.use_bias = use_bias
        self.dtype = dtype

        # init weights (c_hidden, c_hidden)
        if weights is None:
            k = hidden_channels**-0.5
            self.w = Parameter(
                tu.randu((hidden_channels, hidden_channels), -k, k),
                dtype=dtype,
                label="w",
            )
        else:
            self.w = weights

        # init bias (c_out,)
        if use_bias:
            self.b = Parameter(tu.zeros((hidden_channels,)), dtype=dtype, label="w")

    def __repr__(self):
        name = self.__class__.__name__
        hidden_channels = self.hidden_channels
        activation = self.activation
        use_bias = self.use_bias
        dtype = self.dtype
        return f"{name}({hidden_channels=}, {activation=}, {use_bias=}, {dtype=})"

    def __call__(self, x: Tensor) -> Tensor:
        x = x.astype(self.dtype)
        y = tu.zeros_like(x, device=self.device)

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

            def backward(dy: ArrayLike) -> ArrayLike:
                dy = dy.astype(self.dtype)
                self.set_dy(dy)
                dx = tu.zeros_like(x, device=self.device).data
                self.w.grad = tu.zeros_like(self.w, device=self.w.device).data

                for i in range(x.shape[1] - 1, -1, -1):
                    # add hidden state gradient of next layer, if not last sequence element
                    if i == x.shape[1] - 1:
                        out_grad = dy[:, i]
                    else:
                        out_grad = dy[:, i] + dx[:, i + 1] @ self.w.T

                    # activation gradient
                    if self.activation == "tanh":
                        act_grad = -y.data[:, i] ** 2 + 1.0
                    else:
                        act_grad = y.data[:, i] > 0
                    dx[:, i] = act_grad * out_grad

                    # weight grads
                    if i > 0:
                        self.w.grad += y[:, i - 1].T @ dx[:, i]

                # bias grads
                self.b.grad = dx.sum((0, 1))

                return dx

            self.backward = backward

        self.set_y(y)
        return y
