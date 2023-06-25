"""neural network layers module"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import numpy.fft as npfft

from walnut import tensor
from walnut.tensor import Tensor
from walnut.nn import inits, paddings

@dataclass
class Layer:
    """Layer base class."""

    i: int = None
    mode: str = 'eval'
    prev_layer: Layer = None
    succ_layer: Layer = None
    input_shape: tuple[int] = None
    compiled: bool = False
    x: Tensor = None
    y: Tensor = None

    def compile(self, i: int, prev_layer: Layer, succ_layer: Layer) -> None:
        """Connects the layer with adjacent ones and initializes values."""
        if self.prev_layer is None and self.input_shape is None:
            raise AttributeError('input shape must be specified for input layers.')

        self.i = i
        self.prev_layer = prev_layer
        self.succ_layer = succ_layer

        if self.input_shape is not None:
            self.x = tensor.ones((1, *self.input_shape))
        else:
            self.x = Tensor()

        self.y = Tensor()
        self.compiled = True

    def forward(self) -> None:
        """Performs a forward pass."""
        if self.prev_layer is not None:
            self.x.data = self.prev_layer.y.data

    def backward(self) -> None:
        """Performs a backward pass."""
        if self.succ_layer is not None:
            self.y.grad = self.succ_layer.x.grad

@dataclass
class ParamLayer(Layer):
    """Layer using trainable parameters."""
    act_fn: Layer = None
    norm_fn: Layer = None
    init_fn: Callable = inits.kaiming
    use_bias: bool = False
    w: Tensor = None
    b: Tensor = None
    params: list[Tensor] = field(default_factory=list)


class Linear(ParamLayer):
    """Fully connected layer.

    ### Parameters
        out_channels: `int`
            Number of output channels (neurons) of the layer.
        act_fn: `Activation`, optional
            Activation function applied to the layers' output. By default no function is applied.
        norm_fn: `Normalization`
            Normalization applied before activation. By default no normalization is applied.
        init_fn: optional
            Weight initialization method. By default Kaiming He initialization is used.
        use_bias: `bool`, optional
            Whether to use bias values. By default, bias is used.
        input_shape: `tuple[int]`
            Shape of input tensor ignoring axis 0.

    ### Raises
        AttributeError:
            If input shape is not specified for the input layer.
    """

    __slots__ = 'out_channels',

    def __init__(self,
            out_channels: int,
            act_fn: Layer = None,
            norm_fn: Layer = None,
            init_fn: Callable = inits.kaiming,
            use_bias: bool = True,
            input_shape: tuple[int] = None) -> None:
        super().__init__(
            act_fn=act_fn,
            norm_fn=norm_fn,
            init_fn=init_fn,
            use_bias=use_bias,
            input_shape=input_shape)
        self.out_channels = out_channels

    def compile(self, i: int, prev_layer: Layer, succ_layer: Layer) -> None:
        super().compile(i, prev_layer, succ_layer)
        # init weights (c_in, c_out)
        _, in_channels = self.x.shape if self.x.data is not None else self.prev_layer.y.shape
        w_shape = (in_channels, self.out_channels)
        self.w = self.init_fn(shape=w_shape, fan_mode=in_channels, act_fn=self.act_fn)

        # init bias (c_out,)
        if self.use_bias:
            self.b = tensor.zeros((self.out_channels, ))

        self.params = [self.w, self.b]
        self.forward()

    def forward(self) -> None:
        super().forward()
        b = self.b if self.use_bias else 0.0
        self.y.data = (self.x @ self.w + b).data # (b, c_out)

    def backward(self) -> None:
        super().backward()
        self.x.grad = self.y.grad @ self.w.T # input grads (b, c_in)
        self.w.grad = self.x.T @ self.y.grad # weight grads (c_in, c_out)

        if self.use_bias:
            self.b.grad = np.sum(self.y.grad, axis=0) # bias grads (c_out,)


class Convolution(ParamLayer):
    """Convolutional layer used for spacial information and feature extraction.

    ### Parameters
        out_channels: `int`
            Number of output channels (neurons) of the layer.
        kernel_shape: `tuple[int]`, optional
            Shape of each kernel.
        act_fn: `Activation`, optional
            Activation function applied to the layers' output. By default no function is applied.
        norm_fn: `Normalization`
            Normalization applied before activation. By default no normalization is applied.
        init_fn: optional
            Weight initialization method. By default Kaiming He initialization is used.
        pad_fn: optional
            Padding method applied to the input. By default valid padding is used.
        use_bias: `bool`, optional
            Whether to use bias values. By default, bias is used.
        input_shape: `tuple[int]`
            Shape of input tensor ignoring axis 0.

    ### Raises
        AttributeError:
            If input shape is not specified for the input layer.
        ValueError:
            If output shape is smaller than kernel shape.
    """

    __slots__ = 'out_channels', 'kernel_shape', 'pad_fn'

    def __init__(self,
            out_channels: int,
            kernel_shape: tuple[int] = (3, 3),
            act_fn: Layer = None,
            norm_fn: Layer = None,
            init_fn: Callable = inits.kaiming,
            pad_fn: Callable = paddings.valid,
            use_bias: bool = True,
            input_shape: tuple[int] = None) -> None:
        super().__init__(
            act_fn=act_fn,
            norm_fn=norm_fn,
            init_fn=init_fn,
            use_bias=use_bias,
            input_shape=input_shape)
        self.out_channels = out_channels
        self.kernel_shape = kernel_shape
        self.pad_fn = pad_fn

    def compile(self, i: int, prev_layer: Layer, succ_layer: Layer) -> None:
        super().compile(i, prev_layer, succ_layer)
        # init weights (c_out, c_in, y, x)
        _, in_channels, _, _ = self.x.shape if self.x.data is not None else self.prev_layer.y.shape
        w_shape = (self.out_channels, in_channels, *self.kernel_shape)
        _, c_out, w_y, w_x = w_shape
        self.w = self.init_fn(w_shape, fan_mode=c_out*w_y*w_x, act_fn=self.act_fn)

        # init bias (c_out,)
        if self.use_bias:
            self.b = tensor.zeros((self.out_channels,))

        self.params = [self.w, self.b]
        self.forward()
        _, _, y_y, y_x = self.y.shape

        if y_y < w_y or y_x < w_x:
            raise ValueError(self.__class__.__name__, ': Output shape smaller than kernel shape.')

    def forward(self) -> None:
        super().forward()
        x_pad = self.pad_fn(self.x, kernel_shape=self.kernel_shape).data # pad to fit pooling window
        w_rotated = np.flip(self.w.data, axis=(2, 3)) # rotate weights for cross correlation
        x_conv_w = self.__convolve(x_pad, w_rotated, exp_axis=(1, 0)) # (b, _, c_in, y, x) * (_, c_out, c_in, y, x)
        _, _, w_y, w_x = self.w.shape
        self.y.data = np.sum(x_conv_w, axis=2)[:, :, w_y - 1:, w_x - 1:] # sum over input channels

        if self.use_bias:
            batches = self.x.shape[0]
            bias = self.b * tensor.ones((batches, 1)) # broadcast bias over batches (b, c_out)
            self.y.data += tensor.match_dims(bias, 4).data # reshape to fit output (b, c_out, 1, 1)

    def backward(self) -> None:
        super().backward()
        x_p = self.pad_fn(self.x, kernel_shape=self.kernel_shape).data
        _, _, x_y, _ = self.x.shape
        _, _, dy_y, _ = self.y.shape

        # pad grads to fit input after convolution
        if self.pad_fn != paddings.same:
            pad = int((x_y - dy_y) / 2)
            dy_p = np.pad(self.y.grad, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        else:
            dy_p = self.y.grad

        # input grads (b, c_in, y, x)
        dy_conv_w = self.__convolve(dy_p, self.w.data, exp_axis=(2, 0)) # (b, c_out, _, y, x) * (_, c_out, c_in, y, x)
        self.x.grad = np.roll(np.sum(dy_conv_w, axis=1), shift=(-1, -1), axis=(2, 3)) # sum over output channels

        # weight grads (c_out, c_in, y, x)
        dy_conv_x = self.__convolve(x_p, self.y.grad, exp_axis=(1, 2)) # (b, _, c_in, y, x) * (b, c_out, _, y, x)
        _, _, w_y, w_x = self.w.shape
        self.w.grad = np.sum(dy_conv_x, axis=0)[:, :, -w_y:, -w_x:]  # sum over batches

        # bias grads (c_out,)
        if self.use_bias:
            self.b.grad = np.sum(self.y.data, axis=(0, 2, 3)) # sum over batches, y and x

    def __convolve(self, x1: np.ndarray, x2: np.ndarray, exp_axis: tuple[int] = None):
        # fft both tensors
        target_shape = x1.shape[-2:]
        x1_fft = npfft.fft2(x1, s=target_shape)
        x2_fft = npfft.fft2(x2, s=target_shape)

        # expand dims if needed
        if exp_axis:
            ax1, ax2 = exp_axis
            x1_fft_exp = np.expand_dims(x1_fft, ax1)
            x2_fft_exp = np.expand_dims(x2_fft, ax2)

        # multiply, ifft and get real value to complete convolution
        return np.real(npfft.ifft2(x1_fft_exp * x2_fft_exp)).astype('float32')


class MaxPooling(Layer):
    """MaxPoling layer used to reduce information to avoid overfitting.

    ### Parameters
        p_window: `tuple[int]`, optional
            Shape of the pooling window used for the pooling operation.
        input_shape: `tuple[int]`
            Shape of input tensor ignoring axis 0.

    ### Raises
        AttributeError:
            If input shape is not specified for the input layer.
    """

    __slots__ = 'p_window', 'p_map'

    def __init__(self, p_window: tuple[int] = (2, 2), input_shape: tuple[int] = None) -> None:
        super().__init__(input_shape=input_shape)
        self.p_window = p_window
        self.p_map: np.ndarray = None

    def compile(self, i: int, prev_layer: Layer, succ_layer: Layer) -> None:
        super().compile(i, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        # init output as zeros (b, c, y, k)
        x_pad = self.__crop()
        p_y, p_x = self.p_window
        x_b, x_c, _, _ = self.x.shape
        self.y.data = tensor.zeros((x_b, x_c, x_pad.shape[2] // p_y, x_pad.shape[3] // p_x)).data
        self.p_map = tensor.zeros_like(x_pad).data
        _, _, y_y, y_x = self.y.shape

        for y in range(y_y):
            for x in range(y_x):
                chunk = self.x.data[:, :,  y * p_y : (y + 1) * p_y, x * p_x : (x + 1) * p_x]
                self.y.data[:, :, y, x] = np.max(chunk, axis=(2, 3))

        y_s = self.__stretch(self.y.data, self.p_window, (2, 3), x_pad.shape)
        self.p_map = (x_pad == y_s) * 1.0

    def backward(self) -> None:
        super().backward()
        dy_s = self.__stretch(self.y.grad, self.p_window, (2, 3), self.p_map.shape)
        _, _, x_y, x_x = self.x.shape
        self.x.grad = (dy_s * self.p_map)[:, :, :x_y, :x_x] # use p_map as mask for grads

    def __crop(self) -> np.ndarray:
        w_y, w_x = self.p_window
        _, _, x_y, x_x = self.x.shape
        y_fit = x_y // w_y * w_y
        x_fit = x_x // w_x * w_x
        return self.x.data[:, :, :y_fit, :x_fit]

    def __stretch(self,
            x: np.ndarray,
            streching: tuple[int],
            axis: tuple[int],
            target_shape: tuple[int]) -> np.ndarray:
        fa1, fa2 = streching
        ax1, ax2 = axis
        x_stretched = np.repeat(x, fa1, axis=ax1)
        x_stretched = np.repeat(x_stretched, fa2, axis=ax2)
        return np.resize(x_stretched, target_shape)


class Flatten(Layer):
    """Flatten layer used to reshape tensors to shape (b, c_out).

    ### Parameters
        input_shape: `tuple[int]`
            Shape of input tensor ignoring axis 0.

    ### Raises
        AttributeError:
            If input shape is not specified for the input layer.
    """

    def __init__(self, input_shape: tuple[int] = None) -> None:
        super().__init__(input_shape=input_shape)

    def compile(self, i: int, prev_layer: Layer, succ_layer: Layer) -> None:
        super().compile(i, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y.data = self.x.data.reshape(self.x.shape[0], -1)

    def backward(self) -> None:
        super().backward()
        self.x.grad = np.resize(self.y.grad, self.x.shape)


class Dropout(Layer):
    """Dropout layer used to randomly reduce information and avoid overfitting

    ### Parameters
        drop_rate: `float`
            Probability of values being set to 0.
        input_shape: `tuple[int]`
            Shape of input tensor ignoring axis 0.

    ### Raises
        AttributeError:
            If input shape is not specified for the input layer.
    """

    __slots__ = 'd_rate', 'd_map'

    def __init__(self, d_rate: float, input_shape: tuple[int] = None) -> None:
        super().__init__(input_shape=input_shape)
        self.d_rate = d_rate
        self.d_map: np.ndarray = None

    def compile(self, i: int, prev_layer: Layer, succ_layer: Layer) -> None:
        super().compile(i, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()

        if self.mode == 'eval':
            self.y.data = self.x.data
        else:
            d_map = np.random.choice([0, 1], self.x.shape, p=[self.d_rate, 1 - self.d_rate])
            self.d_map = d_map.astype('float32')
            self.y.data = self.x.data * self.d_map / (1 - self.d_rate)

    def backward(self) -> None:
        super().backward()
        self.x.grad = self.y.grad * self.d_map / (1 - self.d_rate) # use d_map as mask for grads
