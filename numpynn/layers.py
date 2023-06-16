# neural network layers module

from numpynn import inits, paddings
import numpy as np
from numpy.fft  import fft2, ifft2
# from numpy.lib.stride_tricks import as_strided


class Layer:
    """Layer base class"""
    
    def __init__(self) -> None:
        self.activation =  self.batch_norm = self.init = None
        self.is_activation_layer = False
        self.has_params = False
        self.compiled = False
        self.mode = 'train'

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

    def compile(self, id: int, prev_layer: object, succ_layer: object) -> None:
        self.id = id
        self.prev_layer = prev_layer
        self.succ_layer = succ_layer
        self.compiled = True

    def forward(self) -> None:
        if self.prev_layer is not None:
            self.x = self.prev_layer.y
    
    def backward(self) -> None:
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

    def compile(self, id: int, prev_layer: object, succ_layer: object) -> None:
        super().compile(id, prev_layer, succ_layer)
        self.x = np.expand_dims(np.ones(self.input_shape), axis=0)
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

    def compile(self, id: int, prev_layer: object, succ_layer: object) -> None:
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

    def __init__(self, nr_neurons: int,  norm=None, activation=None, init=inits.Kaiming, biases=True) -> None:
        super().__init__()
        
        if nr_neurons < 1:
            raise Exception("number of neurons must be at least 1")
        
        self.nr_neurons = nr_neurons
        self.norm = norm
        self.activation = activation
        self.init = init
        self.biases = biases
        self.has_params = True

    def compile(self, id: int, prev_layer: object, succ_layer: object) -> None:
        super().compile(id, prev_layer, succ_layer)

        w_shape = (self.prev_layer.y.shape[1], self.nr_neurons)
        fan_mode = self.prev_layer.y.shape[1]
        self.w = self.init(shape=w_shape, fan_mode=fan_mode, activation=self.activation)
        self.dw = self.w_change = self.w_m = self.w_v = np.zeros_like(self.w)

        if self.biases:
            self.b = self.db = self.b_change = self.b_m = self.b_v = np.zeros((self.nr_neurons,))

        self.forward()

    def forward(self) -> None:
        super().forward()

        self.y = self.x @ self.w + (self.b if self.biases else 0)

    def backward(self) -> None:
        super().backward()

        self.dx = self.dy @ self.w.T
        self.dw = self.x.T @ self.dy

        if self.biases:
            self.db = np.sum(self.dy, axis=0)


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

    def __init__(self, nr_kernels: int, kernel_size: tuple[int, int]=(3, 3), padding=paddings.Valid, norm=None, activation=None, init=inits.Kaiming, biases=True) -> None:       
        super().__init__()

        if nr_kernels < 1:
            raise Exception("number of kernels must be at least 1")

        self.k = nr_kernels
        self.kernel_size = kernel_size
        self.padding = padding
        self.norm = norm
        self.activation = activation
        self.init = init
        self.biases = biases
        self.has_params = True

    def compile(self, id: int, prev_layer: object, succ_layer: object) -> None:
        super().compile(id, prev_layer, succ_layer)
        kernel_shape = (self.k, self.prev_layer.y.shape[1], *self.kernel_size) # (k, c, y, x)
        self.w = self.init(kernel_shape, fan_mode=self.kernel_size[0], activation=self.activation)
        self.dw = self.w_change = self.w_m = self.w_v = np.zeros_like(self.w)
        self.b = self.db = self.b_change = self.b_m = self.b_v = np.zeros((self.k,))
        self.forward()

        if self.y.shape[2] < self.w.shape[2] or self.y.shape[3] < self.w.shape[3]:
            raise Exception(self.__class__.__name__, ': Output shape smaller than kernel shape. Use padding or adjust MaxPooling layer to increase output shape.')
    
    def forward(self) -> None:
        super().forward()
        self.x_p = self.padding(self.x, kernel_size=self.kernel_size)
        Xb, _, Xy, Xx = self.x_p.shape
        Wk, _, Wy, _ = self.w.shape
        self.x_p_fft = fft2(self.x_p)
        w_fft = fft2(np.flip(self.w, axis=(2, 3)), s=(Xy, Xx)) # flip kernel for cross correlation
        p = Wy - 1
        self.y = np.sum(np.real(ifft2(np.expand_dims(self.x_p_fft, 1) * (np.expand_dims(w_fft, 0)))), axis=2)[:, :, p:, p:] # (b, _, c, y, x) + (_, k, c, y, x)
        
        if self.biases:
            self.y += (self.b * np.ones((Xb, 1))).reshape((Xb, Wk, 1, 1))

    def backward(self) -> None:
        super().backward()
        
        if self.padding != paddings.Same:
            p = int((self.x.shape[2] - self.dy.shape[2]) / 2)
            dy_p = np.pad(self.dy, ((0, 0), (0, 0), (p, p), (p, p)))
        else:
            dy_p = self.dy

        _, _, Xy, Xx = self.x_p.shape
        _, _, Wy, Wx = self.w.shape
        _, _, dYpy, dYpx = dy_p.shape

        # dw
        self.dw = np.zeros_like(self.w)
        dy_fft = fft2(self.dy, s=(Xy, Xx))
        self.dw = np.sum(np.real(ifft2(np.expand_dims(dy_fft, 2) * np.expand_dims(self.x_p_fft, 1))), axis=0)[:, :, -Wy:, -Wx:] # (b, k, _, y, x) * (b, _, c, y, x)

        # dx
        self.dx = np.zeros_like(self.x)
        w_fft = fft2(self.w, s=(dYpy, dYpx))
        dy_p_fft = fft2(dy_p)       
        self.dx = np.roll(np.sum(np.real(ifft2(np.expand_dims(dy_p_fft, 2) * np.expand_dims(w_fft, 0))), axis=1), shift=(-1, -1), axis=(2, 3)) # (b, k, _, y, x) * (_, k, c, y, x)

        # db
        if self.biases:
            self.db = np.sum(self.dy, axis=(0, 2, 3))


class MaxPooling(Layer):
    """MaxPoling layer used in convolutional neural network models to reduce information and avoid overfitting.

    Args:
        pooling_window: Size of the pooling window used for the pooling operation.
    """

    def __init__(self, pooling_window: tuple[int, int]) -> None:
        super().__init__()
        self.pooling_window = pooling_window

    def compile(self, id: int, prev_layer: object, succ_layer: object) -> None:
        super().compile(id, prev_layer, succ_layer)
        self.forward()
    
    def forward(self) -> None:
        super().forward()

        wy, wx = self.pooling_window
        b, k, y, x = self.x.shape
        x_pad = self.__pad()

        # faster, but kernel crashes. Have not found the problem yet.
            # stride_bytes = 8 # for float64

            # batch_size, y, x, nr_kernels = x_pad.shape
            # shape = (batch_size, y // 2, x // 2, wy, wx, nr_kernels)

            # b = stride_bytes
            # bk = b * nr_kernels
            # bkx = bk * x
            # bkxb = bkx * batch_size

            # stride = (2 * bkxb, 2 * bkx, 2 * bk , bkx, bk, b) # quite complicated, the docs helped :)

            # s = as_strided(x_pad, shape=shape, strides=stride)
            # self.y = np.max(s, axis=(3, 4)) # <-- kernel crashes here

            # y_r = np.repeat(self.y, wx, axis=1)
            # y_r = np.repeat(y_r, wy, axis=2)
            # self.pooling_map = (x_pad == y_r) * 1.0

        # slower, but stable
        self.y = np.zeros((b, k, x_pad.shape[2] // wy, x_pad.shape[3] // wx))
        self.pooling_map = np.zeros_like(x_pad)

        for y in range(self.y.shape[2]):
            for x in range(self.y.shape[3]):
                array = self.x[:, :,  y * wy : (y + 1) * wy, x * wx : (x + 1) * wx]
                self.y[:, :, y, x] = np.max(array, axis=(2, 3))

        y_r = np.repeat(self.y, wx, axis=2)
        y_r = np.repeat(y_r, wy, axis=3)
        y_r = np.resize(y_r, x_pad.shape)
        self.pooling_map = (x_pad == y_r) * 1.0

    def backward(self) -> None:
        super().backward()

        wy, wx = self.pooling_window
        _, _, y, x = self.x.shape

        dy_r = np.repeat(self.dy, wx, axis=2)
        dy_r = np.repeat(dy_r, wy, axis=3)
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
    "Flatten layer used to reshape tensors with shape (b, x1, x2, ...) into tensors with shape (b, x1 + x2 + ...)."
    
    def __init__(self) -> None:
        super().__init__()

    def compile(self, id: int, prev_layer: object, succ_layer: object) -> None:
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
        self.mode = 'eval'

    def compile(self, id: int, prev_layer: object, succ_layer: object) -> None:
        super().compile(id, prev_layer, succ_layer)
        self.forward()
    
    def forward(self) -> None:
        super().forward()
        if self.mode == 'eval':
            self.y = self.x
        else:
            self.drop_map = np.random.choice([0, 1], self.x.shape, p=[self.drop_rate, 1 - self.drop_rate])
            self.y = self.x * self.drop_map / (1 - self.drop_rate)

    def backward(self) -> None:
        super().backward()
        self.dx = self.dy * self.drop_map / (1 - self.drop_rate)
