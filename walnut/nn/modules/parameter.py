"""parameter modules module"""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import numpy.fft as npfft

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ShapeLike, NumpyArray
from walnut.nn import inits, paddings
from walnut.nn.inits import Init
from walnut.nn.optimizers import Optimizer
from walnut.nn.modules.utility import Module


@dataclass(repr=False, init=False)
class ParamModule(Module):
    """Trainable module base class."""

    def __init__(
        self,
        act_fn_name: str | None = None,
        norm_name: str | None = None,
        init_fn_name: str = "random",
        optimizer: Optimizer | None = None,
        use_bias: bool = True,
        input_shape: ShapeLike | None = None,
    ) -> None:
        super().__init__(input_shape=input_shape)
        self.act_fn_name = act_fn_name
        self.norm_name = norm_name
        self.init_fn_name = init_fn_name
        self.optimizer = optimizer
        self.use_bias = use_bias

        self.w: Tensor = Tensor()
        self.b: Tensor = Tensor()
        self.parameters: list[Tensor] = []

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

    def compile(self, optimizer: Optimizer | None = None) -> None:
        """Connects modules within a model.

        Parameters
        ----------
        optimizer : Optimizer | None, optional
            Optimizer used to update module parameters when training, by default None.
        """
        super().compile()
        self.optimizer = optimizer

    def optimize(self) -> None:
        """Updates module parameters using an optimizer.

        Raises
        ------
        AttributeError
            If no optimizer is defined for the module.
        """
        if self.optimizer:
            for parameter in self.parameters:
                self.optimizer(param=parameter)
        else:
            raise AttributeError("Optimizer not set.")

    def get_parameter_count(self) -> int:
        """Returns the total number of trainable parameters of the module.

        Returns
        -------
        int
            Number of trainable parameters.
        """
        return sum(p.data.size for p in self.parameters)


@dataclass(init=False, repr=False)
class Linear(ParamModule):
    """Fully connected module."""

    def __init__(
        self,
        out_channels: int,
        act: str | None = None,
        norm: str | None = None,
        init: str = "kaiming_he",
        use_bias: bool = True,
        input_shape: ShapeLike | None = None,
    ) -> None:
        """Fully connected module.

        Parameters
        ----------
        out_channels : int
            Number of output channels (neurons) of the module.
        act : str | None, optional
            Activation function applied to the modules outputs, by default None.
        norm : str | None, optional
            Normalization function applied to the modules outputs, by default None.
        init : str, optional
            Initialization function for weights, by default "kaiming_he".
        use_bias : bool, optional
            Whether to use bias values, by default True.
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the module is used as input, by default None.
        """
        super().__init__(
            act_fn_name=act,
            norm_name=norm,
            init_fn_name=init,
            use_bias=use_bias,
            input_shape=input_shape,
        )
        self.out_channels = out_channels
        self.init_fn: Init | None = None
        self.b_sum_axis: ShapeLike | None = None
        self.x_transp_axis: ShapeLike | None = None

    def compile(self, optimizer: Optimizer | None = None) -> None:
        super().compile(optimizer)
        in_channels = self.x.shape[-1]

        # set initializer
        initializer_params = inits.InitParams(in_channels, self.act_fn_name)
        self.init_fn = inits.INITS[self.init_fn_name](initializer_params)

        # init weights (c_in, c_out)
        self.w = self.init_fn((in_channels, self.out_channels))
        self.parameters.append(self.w)

        # init bias (c_out,)
        if self.use_bias:
            self.b = tu.zeros((self.out_channels,))
            self.parameters.append(self.b)

    def forward(self) -> None:
        self.y.data = (self.x @ self.w).data  # (b, [c], c_out)
        if self.use_bias:
            self.y.data += self.b.data

    def backward(self) -> None:
        self.x.grad = self.y.grad @ self.w.T  # input grads (b, c_in)

        # weight grads (c_in, c_out)
        if self.x.ndim == 2:
            self.w.grad = self.x.T @ self.y.grad
        else:
            self.w.grad = np.sum(self.x.transpose((0, 2, 1)).data @ self.y.grad, axis=0)

        # bias grads (c_out,)
        if self.use_bias:
            if self.x.ndim == 2:
                self.b.grad = np.sum(self.y.grad, axis=0)
            else:
                self.b.grad = np.sum(self.y.grad, axis=(0, 1))

        if self.optimizer:
            self.optimize()


@dataclass(init=False, repr=False)
class Convolution(ParamModule):
    """Module used for spacial information and feature extraction."""

    def __init__(
        self,
        out_channels: int,
        kernel_shape: tuple[int, int] = (3, 3),
        act: str | None = None,
        norm: str | None = None,
        init: str = "kaiming_he",
        pad: str = "valid",
        use_bias: bool = True,
        input_shape: ShapeLike | None = None,
    ) -> None:
        """Convolutional module used for spacial information and feature extraction.

        Parameters
        ----------
        out_channels : int
            Number of output channels (neurons) of the module.
        kernel_shape : ShapeLike, optional
            Shape of each kernel, by default (3, 3).
        act : str | None, optional
            Activation function applied to the modules outputs, by default None.
        norm : str | None, optional
            Normalization function applied to the modules outputs, by default None.
        init : str, optional
            Initialization function for weights, by default "kaiming_he".
        pad : str, optional
            Padding method applied to the module imputs, by default "valid".
        use_bias : bool, optional
            Whether to use bias values, by default True.
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the module is used as input, by default None.
        """
        super().__init__(
            act_fn_name=act,
            norm_name=norm,
            init_fn_name=init,
            use_bias=use_bias,
            input_shape=input_shape,
        )
        self.out_channels = out_channels
        self.kernel_shape = kernel_shape
        self.init_fn: Init | None = None
        self.pad_fn_name = pad

        # set padding
        width = math.floor(self.kernel_shape[0] / 2)
        padding_params = paddings.PaddingParams(width, (2, 3))
        self.pad_fn = paddings.PADDINGS[self.pad_fn_name](padding_params)

    def compile(self, optimizer: Optimizer | None = None) -> None:
        super().compile(optimizer)
        in_channels = self.x.shape[1]

        # set initializer
        fan_mode = int(in_channels * np.prod(self.kernel_shape))
        initializer_params = inits.InitParams(fan_mode, self.act_fn_name)
        self.init_fn = inits.INITS[self.init_fn_name](initializer_params)

        # init weights (c_out, c_in, y, x)
        self.w = self.init_fn((self.out_channels, in_channels, *self.kernel_shape))
        self.parameters.append(self.w)

        # init bias (c_out,)
        if self.use_bias:
            self.b = tu.zeros((self.out_channels,))
            self.parameters.append(self.b)

    def forward(self) -> None:
        # apply padding
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
            bias = self.b * tu.ones(shape=(self.x.shape[0], 1))
            # reshape to fit output (b, c_out, 1, 1)
            self.y.data += tu.match_dims(x=bias, dims=4).data

    def backward(self) -> None:
        # input grads (b, c_in, y, x)
        # pad grads to fit input after convolution
        _, _, x_y, _ = self.x.shape
        _, _, dy_y, _ = self.y.shape
        if self.pad_fn_name != "same":
            pad = int((x_y - dy_y) / 2)
            dy_p = np.pad(self.y.grad, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        else:
            dy_p = self.y.grad
        # convolve (b, c_out, _, y, x) * (_, c_out, c_in, y, x)
        dy_conv_w = self.__convolve(dy_p, self.w.data, exp_axis=(2, 0))
        # sum over output channels
        self.x.grad = np.roll(np.sum(dy_conv_w, axis=1), shift=(-1, -1), axis=(2, 3))

        # weight grads (c_out, c_in, y, x)
        x_p = self.pad_fn(self.x).data
        # convolve (b, _, c_in, y, x) * (b, c_out, _, y, x)
        dy_conv_x = self.__convolve(x_p, self.y.grad, exp_axis=(1, 2))
        # sum over batches
        _, _, w_y, w_x = self.w.shape
        self.w.grad = np.sum(dy_conv_x, axis=0)[:, :, -w_y:, -w_x:]

        # bias grads (c_out,)
        if self.use_bias:
            # sum over batches, y and x
            self.b.grad = np.sum(self.y.data, axis=(0, 2, 3))

        if self.optimizer:
            self.optimize()

    # bottleneck
    def __convolve(
        self, x1: NumpyArray, x2: NumpyArray, exp_axis: ShapeLike | None = None
    ) -> NumpyArray:
        # fft both tensors
        target_shape = x1.shape[-2:]
        x1_fft = npfft.fft2(x1, s=target_shape).astype("complex64")
        x2_fft = npfft.fft2(x2, s=target_shape).astype("complex64")

        # expand dims if needed
        if exp_axis:
            ax1, ax2 = exp_axis
            x1_fft = np.expand_dims(x1_fft, ax1)
            x2_fft = np.expand_dims(x2_fft, ax2)

        # multiply, ifft and get real value to complete convolution
        return np.real(npfft.ifft2(x1_fft * x2_fft)).astype("float32")


@dataclass(init=False, repr=False)
class Embedding(ParamModule):
    """Module used for token embedding."""

    def __init__(
        self,
        out_channels: int,
        init: str = "kaiming_he",
        input_shape: ShapeLike | None = None,
    ) -> None:
        """Embedding module used for token embedding.

        Parameters
        ----------
        out_channels : int
            Number of output channels (embedding dimensions) of the module.
        init : str, optional
            Initialization function for weights, by default "kaiming_he".
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the module is used as input, by default None.
        """
        super().__init__(
            init_fn_name=init,
            input_shape=input_shape,
        )
        self.out_channels = out_channels  # embedding dimensions
        self.init_fn: Init | None = None

    def compile(self, optimizer: Optimizer | None = None) -> None:
        super().compile(optimizer)
        vocab_size = self.x.shape[-1]

        # set initializer
        initializer_params = inits.InitParams(vocab_size, self.act_fn_name)
        self.init_fn = inits.INITS[self.init_fn_name](initializer_params)

        # init weights (vocab_size, c_out)
        self.w = self.init_fn((vocab_size, self.out_channels))
        self.parameters.append(self.w)

    def forward(self) -> None:
        self.y.data = (self.x @ self.w).data

    def backward(self) -> None:
        self.w.grad = np.sum(self.x.transpose((0, 2, 1)).data @ self.y.grad, axis=0)
