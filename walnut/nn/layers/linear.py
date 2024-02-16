"""parameter layers layer"""

from __future__ import annotations

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ArrayLike
from walnut.nn.module import Module
from walnut.nn.parameter import Parameter


__all__ = ["Linear"]


class Linear(Module):
    """Fully connected layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        weights: Parameter | None = None,
        use_bias: bool = True,
        dtype: str = "float32",
    ) -> None:
        """Fully connected layer.

        Parameters
        ----------
        in_channels : int
            Number of input channels of the layer.
        out_channels : int
            Number of output channels (neurons) of the layer.
        weights : Parameter | None, optional
            Weights of the layer, by default None. If None, weights are initialized randomly.
        use_bias : bool, optional
            Whether to use bias values, by default True.
        dtype: str, optional
            Datatype of weights and biases, by default "float32".
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.dtype = dtype

        # init weights (c_in, c_out)
        if weights is None:
            k = in_channels**-0.5
            self.w = Parameter(
                tu.randu((in_channels, out_channels), -k, k), dtype=dtype, label="w"
            )
        else:
            self.w = weights

        # init bias (c_out,)
        if use_bias:
            self.b = Parameter(tu.zeros((out_channels,)), dtype=dtype, label="b")

    def __repr__(self) -> str:
        name = self.__class__.__name__
        in_channels = self.in_channels
        out_channels = self.out_channels
        use_bias = self.use_bias
        dtype = self.dtype
        return f"{name}({in_channels=}, {out_channels=}, {use_bias=}, {dtype=})"

    def __call__(self, x: Tensor) -> Tensor:
        x = x.astype(self.dtype)

        y = x @ self.w  # (b, [c], c_out)
        if self.use_bias:
            y += self.b

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                dy = dy.astype(self.dtype)

                # input grads (b, c_in)
                dx = dy @ self.w.T

                # weight grads (c_in, c_out)
                dim = x.ndim
                w_ax = tuple(d if d < dim - 2 else 2 * dim - d - 3 for d in range(dim))
                dw = x.transpose(w_ax).data @ dy
                if dim > 2:
                    wsum_axis = tuple(tu.arange(dim).data[:-2])
                    dw = dw.sum(axis=wsum_axis)

                self.w.grad = dw

                # bias grads (c_out,)
                if self.use_bias:
                    b_tpl = tuple(d for d in range(dim - 1))
                    self.b.grad = dy.sum(axis=b_tpl)

                return dx

            self.backward = backward

        self.set_y(y)
        return y
