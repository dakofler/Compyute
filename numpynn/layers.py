"""neural network layers module"""

import numpy as np
from numpy.fft  import fft2, ifft2
from numpynn import inits, paddings


class Layer:
    """Layer base class"""

    def __init__(self):
        self.compiled = False
        self.mode = 'eval'
        self.i = None
        self.prev_layer = self.succ_layer = None
        self.x = self.y = None
        self.dx = self.dy = None

    def compile(self, i, prev_layer, succ_layer):
        """Connects the layer with adjacent ones."""
        self.i = i
        self.prev_layer = prev_layer
        self.succ_layer = succ_layer
        self.compiled = True

    def forward(self):
        """Performs a forward pass through all layers."""
        if self.prev_layer is not None:
            self.x = self.prev_layer.y

    def backward(self):
        """Performs a backward pass through all layers."""
        if self.succ_layer is not None:
            self.dy = self.succ_layer.dx


class ParamLayer(Layer):
    """Layer using trainable parameters"""

    def __init__(self, init, bias, norm, activation):
        super().__init__()
        self.norm = norm
        self.activation = activation
        self.init = init
        self.bias = bias
        self.dx = None
        self.dy = None
        self.w = self.dw = self.w_delta = self.w_m = self.w_v = None

        if self.bias:
            self.b = self.db = self.b_delta = self.b_m = self.b_v = None


class Input(Layer):
    """Input layer used in neural network models

    Args:
        input_shape: Shape of input tensor ignoring axis 0.
    """

    def __init__(self, input_shape: tuple[int, int]) -> None:
        super().__init__()
        self.input_shape = input_shape

    def compile(self, i, prev_layer, succ_layer) -> None:
        super().compile(i, prev_layer, succ_layer)
        # init with ones and expand by adding batch dim
        self.x = np.expand_dims(inits.ones(self.input_shape), axis=0)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = self.x


class Output(Layer):
    "Output layer used in neural network models"

    def compile(self, i, prev_layer, succ_layer) -> None:
        super().compile(i, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = self.x

    def backward(self) -> None:
        super().backward()
        self.dx = self.dy


class Linear(ParamLayer):
    """Fully connected layer

    Args:
        out_channels: Number of output channels of the layer.
        init: Weight Initialization method [optional].

    Raises:
        Error: If out_channels is less than 1.
    """

    def __init__(self, out_channels: int, norm: Layer = None, activation: Layer = None,
                 init = inits.kaiming, bias = True) -> None:
        super().__init__(init, bias, norm, activation)

        if out_channels < 1:
            raise Exception("out_channels must be >= 1")

        self.out_channels = out_channels

    def compile(self, i, prev_layer, succ_layer) -> None:
        super().compile(i, prev_layer, succ_layer)
        # init weights (c_in, c_out)
        _, prev_out_channels = self.prev_layer.y.shape
        w_shape = (prev_out_channels, self.out_channels)
        self.w = self.init(shape=w_shape, fan_mode=prev_out_channels, activation=self.activation)
        self.dw = self.w_delta = self.w_m = self.w_v = inits.zeros_like(self.w)

        if self.bias:
            # init bias (c_out,)
            self.b = self.db = self.b_delta = self.b_m = self.b_v = inits.zeros((self.out_channels,))

        self.forward()

    def forward(self) -> None:
        super().forward()
        # (b, c_out)
        self.y = self.x @ self.w + (self.b if self.bias else 0)

    def backward(self) -> None:
        super().backward()
        # input gradients (b, c_in)
        self.dx = self.dy @ self.w.T
        # weight gradients (c_in, c_out)
        self.dw = self.x.T @ self.dy

        if self.bias:
            # bias gradients (c_out,)
            self.db = np.sum(self.dy, axis=0)


class Convolution(ParamLayer):
    """Convolutional layer used for spacialinformation

    Args:
        out_channels: Number of output channles of the layer.
        kernel_shape: Dimension of each kernel [optional].
        init: Weight Initialization method [optional].
        padding: Padding applied to the layer's input before processing [optional].

    Raises:
        Error: If the out_channels is less than 1.
    """

    def __init__(self, out_channels: int, kernel_shape: tuple[int, int]=(3, 3),
                 padding=paddings.valid, norm: Layer=None, activation: Layer=None,
                 init=inits.kaiming, bias: bool=True) -> None:
        super().__init__(init, bias, norm, activation)
        
        if out_channels < 1:
            raise Exception("nr_kernels must be >= 1")

        self.out_channels = out_channels
        self.kernel_shape = kernel_shape
        self.padding = padding

    def compile(self, i, prev_layer, succ_layer) -> None:
        super().compile(i, prev_layer, succ_layer)

        # init weights (c_out, c_in, y, x)
        _, prev_out_channels, _, _ = self.prev_layer.y.shape
        w_shape = (self.out_channels, prev_out_channels, *self.kernel_shape)
        _, c_out, w_y, w_x = w_shape
        self.w = self.init(w_shape, fan_mode=c_out*w_y*w_x, activation=self.activation)
        self.dw = self.w_delta = self.w_m = self.w_v = inits.zeros_like(self.w)

        if self.bias:
            # init bias (c_out,)
            self.b = self.db = self.b_delta = self.b_m = self.b_v = inits.zeros((self.out_channels,))

        self.forward()
        _, _, y_y, y_x = self.y.shape

        if y_y < w_y or y_x < w_x:
            raise Exception(self.__class__.__name__, ': Output shape smaller than kernel shape.')

    def forward(self) -> None:
        super().forward()
        # pad input to fit pooling window
        x_pad = self.padding(self.x, kernel_shape=self.kernel_shape)
        x_pad_fft = fft2(x_pad)
        # rotate weights for cross correlation
        w_rotated = np.flip(self.w, axis=(2, 3))
        xp_b, _, xp_y, xp_x = x_pad.shape
        w_fft = fft2(w_rotated, s=(xp_y, xp_x))
        # convolve x (b, _, c_in, y, x) * w (_, c_out, c_in, y, x)
        x_conv_w = np.real(ifft2(np.expand_dims(x_pad_fft, 1) * np.expand_dims(w_fft, 0))).astype('float32')
        # sum over input channels
        w_cout, _, w_y, w_x = self.w.shape
        self.y = np.sum(x_conv_w, axis=2)[:, :, w_y - 1:, w_x - 1:]

        if self.bias:
            # before adding, reshape bias to fit output to get (b, c_out, 1, 1)
            self.y += (self.b * inits.ones((xp_b, 1))).reshape(xp_b, w_cout, 1, 1)

    def backward(self) -> None:
        super().backward()
        x_p = self.padding(self.x, kernel_shape=self.kernel_shape)
        _, _, x_y, _ = self.x.shape
        _, _, dy_y, _ = self.dy.shape

        # pad gradients to fit input after convolution
        if self.padding != paddings.same:
            pad = int((x_y - dy_y) / 2)
            dy_p = np.pad(self.dy, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        else:
            dy_p = self.dy

        # input gradients
        _, _, dyp_y, dyp_x = dy_p.shape
        w_fft = fft2(self.w, s=(dyp_y, dyp_x))
        dy_p_fft = fft2(dy_p)
        # convolve dy (b, c_out, _, y, x) * w (_, c_out, c_in, y, x) and ifft
        dy_conv_w = np.real(ifft2(np.expand_dims(dy_p_fft, 2) * np.expand_dims(w_fft, 0))).astype('float32')
        # sum over output channels
        self.dx = np.roll(np.sum(dy_conv_w, axis=1), shift=(-1, -1), axis=(2, 3))

        # weight gradients
        _, _, xp_y, xp_x = x_p.shape
        dy_fft = fft2(self.dy, s=(xp_y, xp_x))
        x_p_fft = fft2(x_p)
        # convolve dy (b, c_out, _, y, x) * x (b, _, c_in, y, x) and ifft
        dy_conv_x =  np.real(ifft2(np.expand_dims(dy_fft, 2) * np.expand_dims(x_p_fft, 1))).astype('float32')
        # sum over batches
        _, _, w_y, w_x = self.w.shape
        self.dw = np.sum(dy_conv_x, axis=0)[:, :, -w_y:, -w_x:]

        if self.bias:
            # bias gradients
            # sum over batches, y and x
            self.db = np.sum(self.dy, axis=(0, 2, 3))


class MaxPooling(Layer):
    """MaxPoling layer used to reduce information and avoid overfitting

    Args:
        pooling_window: Size of the pooling window used for the pooling operation.
    """

    def __init__(self, pooling_window: tuple[int, int]) -> None:
        super().__init__()
        self.pooling_window = pooling_window
        self.pooling_map = None

    def compile(self, i, prev_layer, succ_layer) -> None:
        super().compile(i, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        # init output as zeros (b, c, y, k)
        x_pad = self.__pad()
        p_y, p_x = self.pooling_window
        x_b, x_c, _, _ = self.x.shape
        self.y = inits.zeros((x_b, x_c, x_pad.shape[2] // p_y, x_pad.shape[3] // p_x))
        self.pooling_map = inits.zeros_like(x_pad)
        _, _, y_y, y_x = self.y.shape

        for y in range(y_y):
            for x in range(y_x):
                # get current chunk
                chunk = self.x[:, :,  y * p_y : (y + 1) * p_y, x * p_x : (x + 1) * p_x]
                # get max value within chunk
                self.y[:, :, y, x] = np.max(chunk, axis=(2, 3))

        # "stretch" output gradients
        y_r = np.repeat(self.y, p_x, axis=2)
        y_r = np.repeat(y_r, p_y, axis=3)
        # resize to fit input
        y_r = np.resize(y_r, x_pad.shape)
        # create pooling map
        # not perfect since technically all values can be equal within a chunk
        self.pooling_map = (x_pad == y_r) * 1.0

    def backward(self) -> None:
        super().backward()
        # stretch gradients
        w_y, w_x = self.pooling_window
        dy_r = np.repeat(self.dy, w_x, axis=2)
        dy_r = np.repeat(dy_r, w_y, axis=3)
        # resize to fit input
        dy_r = np.resize(dy_r, self.pooling_map.shape)
        # use pooling map as mask for gradients
        _, _, x_y, x_x = self.x.shape
        self.dx = (dy_r * self.pooling_map)[:, :, :x_y, :x_x]

    def __pad(self):
        w_y, w_x = self.pooling_window
        _, _, x_y, x_x = self.x.shape
        y_delta = (w_y - x_y % w_y) % w_y
        x_delta = (w_x - x_x % w_x) % w_x
        return np.pad(self.x, ((0, 0), (0, 0), (0, y_delta), (0, x_delta)))


class Flatten(Layer):
    "Flatten layer used to reshape tensors to shape (b, c_out)"

    def compile(self, i, prev_layer, succ_layer) -> None:
        super().compile(i, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = self.x.reshape(self.x.shape[0], -1)

    def backward(self) -> None:
        super().backward()
        self.dx = np.resize(self.dy, self.x.shape)


class Dropout(Layer):
    """Dropout layer used to randomly reduce information and avoid overfitting

    Args:
        drop_rate: Probability of values being set to 0.

    Raises:
        Error: If the droprate is outside the interval [0, 1).
    """

    def __init__(self, drop_rate: float) -> None:
        super().__init__()

        if drop_rate < 0 or drop_rate >= 1:
            raise Exception("drop rate must be in the interval [0, 1)")

        self.drop_rate = drop_rate
        self.drop_map = None

    def compile(self, i, prev_layer, succ_layer) -> None:
        super().compile(i, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()

        if self.mode == 'eval':
            self.y = self.x
        else:
            self.drop_map = np.random.choice([0, 1], self.x.shape, p=[self.drop_rate, 1 - self.drop_rate]).astype('float32')
            self.y = self.x * self.drop_map / (1 - self.drop_rate)

    def backward(self) -> None:
        super().backward()
        self.dx = self.dy * self.drop_map / (1 - self.drop_rate)
