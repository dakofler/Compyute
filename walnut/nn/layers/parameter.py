"""parameter layers module"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import numpy.fft as npfft

from walnut import tensor
from walnut.tensor import Tensor
from walnut.nn.inits import Init
from walnut.nn.paddings import Padding
from walnut.nn.optimizers import Optimizer
from walnut.nn.layers.utility import Layer


@dataclass()
class ParamLayer(Layer):
    """Trainable layer base class."""

    optimizer: Optimizer | None = None
    init_fn: Init | None = None
    use_bias: bool = False
    w: Tensor = Tensor()
    b: Tensor = Tensor()
    parameters: list[Tensor] = field(default_factory=list)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        x_shape = str(self.x.shape[1:])
        w_shape = str(self.w.shape)
        b_shape = str(self.b.shape)
        y_shape = str(self.y.shape[1:])
        params = str(sum(p.data.size for p in self.parameters))
        return (
            f"{name:15s} | {x_shape:15s} | {w_shape:15s} | "
            + f"{b_shape:15s} | {y_shape:15s} | {params:15s}"
        )

    def optimize(self) -> None:
        """Updates layer parameters using an optimizer object."""
        if self.optimizer:
            for parameter in self.parameters:
                self.optimizer(parameter=parameter)

    def get_parameter_count(self) -> int:
        """Returns the total number of trainable parameters of the layer.

        Returns
        -------
        int
            Number of trainable parameters.
        """
        return sum(p.data.size for p in self.parameters)


@dataclass(init=False, repr=False)
class Linear(ParamLayer):
    """Fully connected layer."""

    def __init__(
        self,
        out_channels: int,
        init_fn: Init,
        use_bias: bool = True,
        input_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Fully connected layer.

        Parameters
        ----------
        out_channels : int
            Number of output channels (neurons) of the layer.
        init_fn : Callable[..., Tensor], optional
            Weight initialization method, by default inits.kaiming.
        use_bias : bool, optional
            Whether to use bias values, by default True.
        input_shape : tuple[int] | None, optional
            Shape of a sample. Required if the layer is used as input, by default None.
        """
        super().__init__(
            init_fn=init_fn,
            use_bias=use_bias,
            input_shape=input_shape,
        )
        self.out_channels = out_channels

    def compile(self) -> None:
        super().compile()
        # init weights (c_in, c_out)
        _, in_channels = (
            self.prev_layer.y.shape if self.prev_layer is not None else self.x.shape
        )
        w_shape = (in_channels, self.out_channels)
        if self.init_fn is not None:
            self.w = self.init_fn(w_shape)

        # init bias (c_out,)
        if self.use_bias:
            self.b = tensor.zeros((self.out_channels,))

        self.parameters = [self.w, self.b]

    def forward(self, mode: str = "eval") -> None:
        super().forward()
        bias = self.b if self.use_bias else 0.0
        self.y.data = (self.x @ self.w + bias).data  # (b, c_out)

    def backward(self) -> None:
        super().backward()
        self.x.grad = self.y.grad @ self.w.T  # input grads (b, c_in)
        self.w.grad = self.x.T @ self.y.grad  # weight grads (c_in, c_out)
        if self.use_bias:
            self.b.grad = np.sum(self.y.grad, axis=0)  # bias grads (c_out,)
        self.optimize()


@dataclass(init=False, repr=False)
class Convolution(ParamLayer):
    """Convolutional layer used for spacial information and feature extraction."""

    def __init__(
        self,
        out_channels: int,
        init_fn: Init,
        pad_fn: Padding,
        kernel_shape: tuple[int, int] = (3, 3),
        use_bias: bool = True,
        input_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Convolutional layer used for spacial information and feature extraction.

        Parameters
        ----------
        out_channels : int
            Number of output channels (neurons) of the layer.
        kernel_shape : tuple[int, int], optional
            Shape of each kernel, by default (3, 3).
        init_fn : Callable[..., Tensor], optional
            Weight initialization method, by default inits.kaiming.
        pad_fn : Callable[..., Tensor], optional
            Padding method applied to the input, by default paddings.valid.
        use_bias : bool, optional
            Whether to use bias values, by default True.
        input_shape : tuple[int, ...] | None, optional
            Shape of a sample. Required if the layer is used as input, by default None.
        """
        super().__init__(
            init_fn=init_fn,
            use_bias=use_bias,
            input_shape=input_shape,
        )
        self.out_channels = out_channels
        self.kernel_shape = kernel_shape
        self.pad_fn = pad_fn

    def compile(self) -> None:
        super().compile()
        # init weights (c_out, c_in, y, x)
        _, in_channels, _, _ = (
            self.prev_layer.y.shape if self.prev_layer is not None else self.x.shape
        )
        w_shape = (self.out_channels, in_channels, *self.kernel_shape)
        if self.init_fn is not None:
            self.w = self.init_fn(w_shape)

        # init bias (c_out,)
        if self.use_bias:
            self.b = tensor.zeros((self.out_channels,))

        self.parameters = [self.w, self.b]

    def forward(self, mode: str = "eval") -> None:
        super().forward()
        # pad to fit pooling window
        x_pad = self.pad_fn(self.x).data
        # rotate weights for cross correlation
        w_rotated = np.flip(self.w.data, axis=(2, 3))
        # convolve (b, _, c_in, y, x) * (_, c_out, c_in, y, x)
        x_conv_w = self.__convolve(x_pad, w_rotated, exp_axis=(1, 0))
        # sum over input channels
        _, _, w_y, w_x = self.w.shape
        self.y.data = np.sum(x_conv_w, axis=2)[:, :, w_y - 1 :, w_x - 1 :]
        if self.use_bias:
            # broadcast bias over batches (b, c_out)
            bias = self.b * tensor.ones(shape=(self.x.shape[0], 1))
            # reshape to fit output (b, c_out, 1, 1)
            self.y.data += tensor.match_dims(x=bias, dims=4).data

    def backward(self) -> None:
        super().backward()
        x_p = self.pad_fn(self.x).data
        _, _, x_y, _ = self.x.shape
        _, _, dy_y, _ = self.y.shape

        # pad grads to fit input after convolution
        # TODO: Find more elegant solution
        if self.pad_fn.__class__.__name__ != "Same":
            pad = int((x_y - dy_y) / 2)
            dy_p = np.pad(self.y.grad, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        else:
            dy_p = self.y.grad

        # input grads (b, c_in, y, x)
        # convolve (b, c_out, _, y, x) * (_, c_out, c_in, y, x)
        dy_conv_w = self.__convolve(dy_p, self.w.data, exp_axis=(2, 0))
        # sum over output channels
        self.x.grad = np.roll(np.sum(dy_conv_w, axis=1), shift=(-1, -1), axis=(2, 3))

        # weight grads (c_out, c_in, y, x)
        # convolve (b, _, c_in, y, x) * (b, c_out, _, y, x)
        dy_conv_x = self.__convolve(x_p, self.y.grad, exp_axis=(1, 2))
        # sum over batches
        _, _, w_y, w_x = self.w.shape
        self.w.grad = np.sum(dy_conv_x, axis=0)[:, :, -w_y:, -w_x:]

        # bias grads (c_out,)
        if self.use_bias:
            # sum over batches, y and x
            self.b.grad = np.sum(self.y.data, axis=(0, 2, 3))

        self.optimize()

    def __convolve(
        self, x1: np.ndarray, x2: np.ndarray, exp_axis: tuple[int, ...] | None = None
    ):
        # fft both tensors
        target_shape = x1.shape[-2:]
        x1_fft = npfft.fft2(x1, s=target_shape)
        x2_fft = npfft.fft2(x2, s=target_shape)

        # expand dims if needed
        if exp_axis:
            ax1, ax2 = exp_axis
            x1_fft = np.expand_dims(x1_fft, ax1)
            x2_fft = np.expand_dims(x2_fft, ax2)

        # multiply, ifft and get real value to complete convolution
        return np.real(npfft.ifft2(x1_fft * x2_fft)).astype("float32")
