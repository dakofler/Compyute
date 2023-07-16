"""parameter modules module"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ShapeLike, NumpyArray
from walnut.nn import inits
from walnut.nn.inits import Init
from walnut.nn.optimizers import Optimizer
from walnut.nn.funcional import convolve2d
from walnut.nn.modules.module import Module


__all__ = ["Linear", "Convolution2d", "Embedding"]


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

        self.w: Tensor = tu.empty()
        self.b: Tensor = tu.empty()
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

    def __call__(self, x: Tensor) -> Tensor:
        super().__call__(x)
        self.y.data = (self.x @ self.w).data  # (b, [c], c_out)
        if self.use_bias:
            self.y.data += self.b.data
        return self.y

    def backward(self, y_grad: NumpyArray) -> NumpyArray:
        super().backward(y_grad)
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

        return self.x.grad


@dataclass(init=False, repr=False)
class Convolution2d(ParamModule):
    """Module used for spacial information and feature extraction."""

    def __init__(
        self,
        out_channels: int,
        kernel_shape: tuple[int, int] = (3, 3),
        act: str | None = None,
        norm: str | None = None,
        init: str = "kaiming_he",
        stride: int = 1,
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
        stride : int, optional
            Stride used for the convolution operation, by default 1.
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
        self.stride = stride

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

    def __call__(self, x: Tensor) -> Tensor:
        super().__call__(x)
        # rotate weights for cross correlation
        w_rotated = self.w.flip(axis=(2, 3))

        # convolve (b, _, c_in, y, x) * (_, c_out, c_in, y, x)
        x_ext = tu.expand_dims(self.x, 1)  # add fake c_out dim
        w_rotated_ext = tu.expand_dims(w_rotated, 0)  # add fake b dim
        x_conv_w = convolve2d(x_ext, w_rotated_ext, self.stride)

        # sum over input channels
        self.y.data = x_conv_w.sum(axis=2).data

        if self.use_bias:
            # broadcast bias over batches (b, c_out)
            bias = self.b * tu.ones(shape=(self.x.shape[0], 1))
            # reshape to fit output (b, c_out, 1, 1)
            self.y.data += tu.match_dims(x=bias, dims=4).data

        return self.y

    def backward(self, y_grad: NumpyArray) -> NumpyArray:
        super().backward(y_grad)

        # undo strides by filling with zeros
        b, c, dy_y, dy_x = self.y.grad.shape
        y_grad_p = np.zeros((b, c, self.stride * dy_y, self.stride * dy_x))
        y_grad_p[:, :, :: self.stride, :: self.stride] = self.y.grad
        out_y = 1 + (self.x.shape[-2] - self.w.shape[-2])
        out_x = 1 + (self.x.shape[-1] - self.w.shape[-1])
        y_grad_p = y_grad_p[:, :, :out_y, :out_x]

        # input grads (b, c_in, y, x)
        y_grad_p_ext = tu.expand_dims(Tensor(y_grad_p), 2)
        w_ext = tu.expand_dims(self.w, 0)
        # convolve (b, c_out, _, y, x) * (_, c_out, c_in, y, x)
        dy_conv_w = convolve2d(y_grad_p_ext, w_ext, mode="full")
        self.x.grad = dy_conv_w.sum(axis=1).data  # sum over output channels

        # weight grads (c_out, c_in, y, x)
        x_ext = tu.expand_dims(self.x, 1)
        y_grad_p_ext = y_grad_p_ext.flip((-2, -1))
        # convolve (b, _, c_in, y, x) * (b, c_out, _, y, x)
        x_conv_dy = convolve2d(x_ext, y_grad_p_ext)
        self.w.grad = x_conv_dy.sum(axis=0).data  # sum over batches

        # bias grads (c_out,)
        if self.use_bias:
            self.b.grad = np.sum(self.y.grad, axis=(0, 2, 3))  # sum over b, y and x

        if self.optimizer:
            self.optimize()
        return self.x.grad


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

    def __call__(self, x: Tensor) -> Tensor:
        super().__call__(x)
        self.y.data = (self.x @ self.w).data
        return self.y

    def backward(self, y_grad: NumpyArray) -> NumpyArray:
        super().backward(y_grad)
        self.w.grad = np.sum(self.x.transpose((0, 2, 1)).data @ self.y.grad, axis=0)
        return self.x.grad
