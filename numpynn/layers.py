# neural network layers module

from numpynn import inits, paddings, utils
import numpy as np
from numpy.fft  import fft2, ifft2
# from numpy.lib.stride_tricks import as_strided


class Layer:
    """Layer base class"""
    
    def __init__(self):
        self.norm = self.activation = self.init = None
        self.is_activation_layer = False
        self.compiled = False
        self.mode = 'eval'

        self.prev_layer = None
        self.succ_layer = None
        self.x = None
        self.dx = None
        self.y = None
        self.dy = None
        self.w = None
        self.w_change = None
        self.dw = None
        self.b = None
        self.b_change = None
        self.db = None

    def compile(self, id, prev_layer, succ_layer):
        self.id = id
        self.prev_layer = prev_layer
        self.succ_layer = succ_layer
        self.compiled = True

    def forward(self):
        if self.prev_layer is not None:
            self.x = self.prev_layer.y
    
    def backward(self):
        if self.succ_layer is not None:
            self.dy = self.succ_layer.dx


class Input(Layer):
    """Input layer used in neural network models.

    Args:
        input_shape: Shape of input arrays the model is trained and/or applied to.
    """
    
    def __init__(self, input_shape: tuple[int, int]) -> None:
        super().__init__()
        self.input_shape = input_shape

    def compile(self, id, prev_layer, succ_layer):
        super().compile(id, prev_layer, succ_layer)
        # init with ones and expand by adding batch dim
        self.x = np.expand_dims(inits.ones(self.input_shape), axis=0)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = self.x
    
    def backward(self) -> None:
        super().backward()
    

class Output(Layer):
    "Output layer used in neural network models."

    def __init__(self) -> None:
        super().__init__()

    def compile(self, id, prev_layer, succ_layer):
        super().compile(id, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = self.x
    
    def backward(self) -> None:
        super().backward()
        self.dx = self.dy


class Linear(Layer):
    """Fully connected layer used in neural network models.

    Args:
        nr_neurons: Number of neurons to be used in this layer.
        init: Weight Initialization method [optional].

    Raises:
        Error: If the number of neurons is less than 1.
    """

    def __init__(self, nr_neurons: int, norm: Layer = None, activation: Layer = None,
                 init = inits.kaiming, bias = True) -> None:
        super().__init__()

        if nr_neurons < 1:
            raise Exception("number of neurons must be at least 1")

        self.nr_neurons = nr_neurons
        self.norm = norm
        self.activation = activation
        self.init = init
        self.bias = bias

    def compile(self, id, prev_layer, succ_layer):
        super().compile(id, prev_layer, succ_layer)
        _, X = self.prev_layer.y.shape
        w_shape = (X, self.nr_neurons)
        self.w = self.init(shape=w_shape, fan_mode=X, activation=self.activation) # (X, N)
        self.dw = self.w_change = self.w_m = self.w_v = inits.zeros_like(self.w) # (X, N)
        
        if self.bias:
            # init bias and bias gradients
            self.b = self.db = self.b_change = self.b_m = self.b_v = inits.zeros((self.nr_neurons,), dtype='float32') # (N,)

        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = self.x @ self.w + (self.b if self.bias else 0) # (B, N)

    def backward(self) -> None:
        super().backward()
        # input gradients
        self.dx = self.dy @ self.w.T # (B, X)
        # weight gradients
        self.dw = self.x.T @ self.dy # (X, N)

        # bias gradients
        if self.bias:
            self.db = np.sum(self.dy, axis=0) # (N,)


class Convolution(Layer):
    """Convolutional layer used in convolutional neural network models to extract features from images.

    Args:
        nr_kernels: Number of kernels to be used in this layer.
        kernel_size: Size of each kernel [optional].
        padding: Padding function that is applied to the layer's input before processing [optional].
        init: Weight Initialization method [optional].

    Raises:
        Error: If the number of kernels is less than 1.
    """

    def __init__(self, nr_kernels: int, kernel_size: tuple[int, int]=(3, 3),
                 padding=paddings.valid, norm: Layer=None, activation: Layer=None,
                 init=inits.kaiming, bias: bool=True) -> None:       
        super().__init__()
        
        if nr_kernels < 1:
            raise Exception("Number of kernels must be at least 1")

        self.k = nr_kernels
        self.kernel_size = kernel_size
        self.padding = padding
        self.norm = norm
        self.activation = activation
        self.init = init
        self.bias = bias

    def compile(self, id, prev_layer, succ_layer):
        super().compile(id, prev_layer, succ_layer)
        w_shape = (self.k, self.prev_layer.y.shape[1], *self.kernel_size)
        _, C, Wy, Wx = w_shape

        self.w = self.init(w_shape, fan_mode=C*Wy*Wx, activation=self.activation) # (K, C, Y, X)
        self.dw = self.w_change = self.w_m = self.w_v = inits.zeros_like(self.w) # (K, C, Y, X)

        if self.bias:
            self.b = self.db = self.b_change = self.b_m = self.b_v = inits.zeros((self.k,)) # (K, )

        self.forward()
        _, _, Yy, Yx = self.y.shape

        if Yy < Wy or Yx < Wx:
            raise Exception(self.__class__.__name__, ': Output shape smaller than kernel shape.')

    def forward(self) -> None:
        super().forward()
        x_p = self.padding(self.x, kernel_size=self.kernel_size)
        Xb, _, Xy, Xx = x_p.shape
        Wk, _, Wy, _ = self.w.shape

        x_p_fft = fft2(x_p)
        # rotate weights for cross correlation
        w_rotated = np.flip(self.w, axis=(2, 3))
        w_fft = fft2(w_rotated, s=(Xy, Xx))
        # convolve x * w
        x_conv_w = np.real(ifft2(np.expand_dims(x_p_fft, 1) * np.expand_dims(w_fft, 0))).astype('float32') # (B, _, C, Y, X) * (_, K, C, Y, X)
        # sum over input channels
        self.y = np.sum(x_conv_w, axis=2)[:, :, Wy - 1:, Wy - 1:] # (B, K, Y, X)

        if self.bias:
            # reshape bias to fit output
            b = (self.b * inits.ones((Xb, 1))).reshape(Xb, Wk, 1, 1) # (B, K, 1, 1)
            self.y += b

    def backward(self) -> None:
        super().backward()
        x_p = self.padding(self.x, kernel_size=self.kernel_size)
        _, _, Xy, _ = self.x.shape
        _, _, dYy, _ = self.dy.shape
        _, _, Xpy, Xpx = x_p.shape
        _, _, Wy, Wx = self.w.shape

        # pad gradients if necessary
        if self.padding != paddings.same:
            p = int((Xy - dYy) / 2)
            dy_p = np.pad(self.dy, ((0, 0), (0, 0), (p, p), (p, p)))
        else:
            dy_p = self.dy

        _, _, dYpy, dYpx = dy_p.shape

        # weight gradients
        dy_fft = fft2(self.dy, s=(Xpy, Xpx))
        x_p_fft = fft2(x_p)
        # convolve dy * x and ifft
        dy_conv_x =  np.real(ifft2(np.expand_dims(dy_fft, 2) * np.expand_dims(x_p_fft, 1))).astype('float32') # (B, K, _, Y, X) * (B, _, C, Y, X)
        # sum over batches
        self.dw = np.sum(dy_conv_x, axis=0)[:, :, -Wy:, -Wx:] # (K, C, Y, X)

        # input gradients
        w_fft = fft2(self.w, s=(dYpy, dYpx))
        dy_p_fft = fft2(dy_p)
        # convolve dy * w and ifft
        dy_conv_w = np.real(ifft2(np.expand_dims(dy_p_fft, 2) * np.expand_dims(w_fft, 0))).astype('float32') # (B, K, _, Y, X) * (_, K, C, Y, X)
        # sum over kernels
        self.dx = np.roll(np.sum(dy_conv_w, axis=1), shift=(-1, -1), axis=(2, 3)) # (B, C, Y, X)

        # bias gradients
        # sum over batches, y and x
        if self.bias:
            self.db = np.sum(self.dy, axis=(0, 2, 3)) # (K,)


class MaxPooling(Layer):
    """MaxPoling layer used in convolutional neural network models to reduce information and avoid overfitting.

    Args:
        pooling_window: Size of the pooling window used for the pooling operation.
    """

    def __init__(self, pooling_window: tuple[int, int]) -> None:
        super().__init__()
        self.pooling_window = pooling_window

    def compile(self, id, prev_layer, succ_layer):
        super().compile(id, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        Py, Px = self.pooling_window
        Xb, Xk, _, _ = self.x.shape
        x_pad = self.__pad()
        self.y = inits.zeros((Xb, Xk, x_pad.shape[2] // Py, x_pad.shape[3] // Px)) # (B, K, Y, X)
        self.pooling_map = inits.zeros_like(x_pad)
        _, _, Yy, Yx = self.y.shape

        for y in range(Yy):
            for x in range(Yx):
                # get current chunk
                array = self.x[:, :,  y * Py : (y + 1) * Py, x * Px : (x + 1) * Px]
                # get max value within chunk
                self.y[:, :, y, x] = np.max(array, axis=(2, 3))

        # stretch gradients
        y_r = np.repeat(self.y, Px, axis=2)
        y_r = np.repeat(y_r, Py, axis=3)
        # resize to fit input
        y_r = np.resize(y_r, x_pad.shape)
        # not perfect since technically all values can be equal within a chunk
        self.pooling_map = (x_pad == y_r) * 1.0

    def backward(self) -> None:
        super().backward()
        wy, wx = self.pooling_window
        _, _, y, x = self.x.shape

        # stretch gradients
        dy_r = np.repeat(self.dy, wx, axis=2)
        dy_r = np.repeat(dy_r, wy, axis=3)
        # resize to fit input
        dy_r = np.resize(dy_r, self.pooling_map.shape)
        dx = dy_r * self.pooling_map
        self.dx = dx[:, :, :y, :x]
    
    def __pad(self) -> np.ndarray:
        wy, wx = self.pooling_window
        _, _, y, x = self.x.shape
        dy = (wy - y % wy) % wy
        dx = (wx - x % wx) % wx
        return np.pad(self.x, ((0, 0), (0, 0), (0, dy), (0, dx)))


class Flatten(Layer):
    "Flatten layer used to reshape tensors to shape (B, N)."
    
    def __init__(self) -> None:
        super().__init__()

    def compile(self, id, prev_layer, succ_layer):
        super().compile(id, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        super().forward()
        self.y = self.x.reshape(self.x.shape[0], -1)
    
    def backward(self) -> None:
        super().backward()
        self.dx = np.resize(self.dy, self.x.shape)


class Dropout(Layer):
    """Dropout layer used in neural network models to reduce information and avoid overfitting.

    Args:
        drop_rate: Probability of values of the input being set to 0.

    Raises:
        Error: If the droprate is outside the interval [0, 1).
    """

    def __init__(self, drop_rate: float) -> None:       
        super().__init__()

        if drop_rate < 0 or drop_rate >= 1:
            raise Exception("drop rate must be in the interval [0, 1)")
        
        self.drop_rate = drop_rate

    def compile(self, id, prev_layer, succ_layer):
        super().compile(id, prev_layer, succ_layer)
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
