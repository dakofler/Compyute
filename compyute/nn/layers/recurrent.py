"""recurrent layers layer"""

from compyute.functional import random_uniform, zeros, zeros_like
from compyute.nn.module import Module
from compyute.nn.parameter import Parameter
from compyute.tensor import Tensor, ArrayLike


__all__ = ["RecurrentCell"]


class RecurrentCell(Module):
    """Recurrent layer."""

    def __init__(
        self,
        hidden_channels: int,
        weights: Parameter | None = None,
        use_bias: bool = True,
        dtype: str = "float32",
    ) -> None:
        """Recurrent layer.

        Parameters
        ----------
        hidden_channels : int
            Number of hidden channels of the layer.
        weights : Parameter | None, optional
            Weights of the layer, by default None. If None, weights are initialized randomly.
        use_bias : bool, optional
            Whether to use bias values, by default True.
        dtype: str, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.use_bias = use_bias
        self.dtype = dtype

        # init weights (c_hidden, c_hidden)
        if weights is None:
            k = hidden_channels**-0.5
            w = random_uniform((hidden_channels, hidden_channels), -k, k)
            self.w = Parameter(w, dtype=dtype, label="w")
        else:
            self.w = weights

        # init bias (c_out,)
        if use_bias:
            self.b = Parameter(zeros((hidden_channels,)), dtype=dtype, label="b")

    def __repr__(self):
        name = self.__class__.__name__
        hidden_channels = self.hidden_channels
        use_bias = self.use_bias
        dtype = self.dtype
        return f"{name}({hidden_channels=}, {use_bias=}, {dtype=})"

    def forward(self, x: Tensor) -> Tensor:
        self.check_dims(x, [3])
        x = x.astype(self.dtype)
        y = zeros_like(x, device=self.device)

        # iterate over sequence elements
        for i in range(x.shape[1]):
            # hidden states
            h = y[:, i - 1] @ self.w
            if self.use_bias:
                h += self.b

            # activation
            y[:, i] = (x[:, i] + h).tanh()

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                dy = dy.astype(self.dtype)
                self.set_dy(dy)

                dx = zeros_like(x, device=self.device).data
                self.w.grad = zeros_like(self.w, device=self.w.device).data

                for i in range(x.shape[1] - 1, -1, -1):
                    # add hidden state gradient of next layer, if not last sequence element
                    if i == x.shape[1] - 1:
                        out_grad = dy[:, i]
                    else:
                        out_grad = dy[:, i] + dx[:, i + 1] @ self.w.T

                    # activation gradient
                    act_grad = 1 - y.data[:, i] ** 2
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
