from model import activations, paddings, utils
import numpy as np
from numpy.fft  import fft2, ifft2
import math

class Layer:
    def __init__(self):
        self.name = 'layer'
        self.i = None
        self.o = None
        self.input_shape = None
        self.w = None
        self.b = None
        self.pooling_window = None
        self.k = None
        self.kernel_size = None
        self.dy = None
        self.dx = None
        self.dw = None
        self.db = None
        self.w_change = None
        self.b_change = None
        self.drop_rate = None

    def integrate(self, id, prev_layer, succ_layer):
        self.id = id
        self.prev_layer = prev_layer
        self.succ_layer = succ_layer

    def process(self):
        if self.prev_layer is not None:
            self.i = self.prev_layer.o
    
    def learn(self):
        if self.succ_layer is not None:
            self.dy = self.succ_layer.dx


class Input(Layer):
    def __init__(self, input_shape) -> None:
        super().__init__()
        self.name = 'input'
        self.input_shape = input_shape

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.i = np.ones(self.input_shape)
        self.process()
        self.summary = ''

    def process(self):
        super().process()
        self.o = np.reshape(self.i, self.input_shape)
    
    def learn(self):
        super().learn()
    

class Output(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'output'

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.process()
        self.summary = ''

    def process(self):
        super().process()
        self.o = self.i
    
    def learn(self):
        super().learn()
        self.dx = self.dy


class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'flatten'

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.process()
        self.summary = f'{self.name}\t\t{str(self.i.shape)}\t{str(self.o.shape)}\t\t0'

    def process(self):
        super().process()
        self.o = self.i.flatten()
    
    def learn(self):
        super().learn()
        self.dx = np.resize(self.dy, self.i.shape)


class Dense(Layer):
    def __init__(self, nr_neurons, activation=activations.Identity) -> None:
        super().__init__()
        self.name = 'dense'
        self.nr_neurons = nr_neurons
        self.activation = activation

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.w = np.random.uniform(-1.0, 1.0, (self.nr_neurons, self.prev_layer.o.shape[0] + 1)) # +1 for the bias neuron weights
        self.dw = np.zeros(self.w.shape)
        self.w_change = np.zeros(self.w.shape)
        self.process()
        self.summary = f'{self.name}\t\t{str(self.i.shape)}\t\t{str(self.o.shape)}\t\t{str(np.size(self.w))}'

    def process(self):
        super().process()
        input = np.append(self.i, [1.0], axis=0) # add bias neuron
        self.net = np.dot(self.w, input)
        self.o = self.activation(self.net)

    def learn(self):
        super().learn()

        # derivative of activation function
        if self.activation == activations.Softmax: # https://e2eml.school/softmax.html
            d_softmax = self.activation(self.net, derivative=True)
            self.dy = np.reshape(self.dy, (1, -1))
            dy = np.squeeze(self.dy @ d_softmax)
        else:
            dy = self.activation(self.net, derivative=True) * self.dy

        w = np.delete(self.w.copy(), -1, axis=1) # remove weights corresponding to bias neurons to make shapes match
        self.dx = w.T @ dy
        self.dw = np.append(self.prev_layer.o, [1.0], axis=0) * np.expand_dims(dy, 1)


class MaxPooling(Layer):
    def __init__(self, pooling_window) -> None:
        super().__init__()
        self.name = 'maxpooling'
        self.pooling_window = pooling_window

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.process()
        self.summary = f'{self.name}\t{str(self.i.shape)}\t{str(self.o.shape)}\t0'
    
    def process(self):
        super().process()
        i_y, i_x, i_k  = self.i.shape
        step = self.pooling_window[0]
        o_x = int(i_x / step)
        o_y = int(i_y / step)

        self.o = np.zeros((o_y, o_x, i_k))
        self.pooling_map = np.zeros(self.i.shape)

        for k in range(i_k):
            image = self.i[:, :, k]
            for y in range(o_y):
                for x in range(o_x):
                    a = image[y * step : y * step + step, x * step : x * step + step] # get sub matrix
                    index = np.where(a == np.max(a)) # get index of max value in sub matrix
                    index_y = index[0][0] + y * step # get y index of max value in input matrix
                    index_x = index[1][0] + x * step # get x index of max value in input matrix

                    self.o[y, x, k] = image[index_y, index_x]
                    self.pooling_map[index_y, index_x, k] = 1

    def learn(self):
        super().learn()
        dy = np.repeat(self.dy, self.pooling_window[0], axis=0)
        dy = np.repeat(dy, self.pooling_window[0], axis=1)
        dy = np.resize(dy, self.pooling_map.shape) # if input mod pooling size is not 0, gradient and map shape is not equal. Resize fills missing values with 0.
        self.dx = dy * self.pooling_map


class Convolution(Layer):
    def __init__(self, nr_kernels, kernel_size=(3, 3), activation=activations.Identity, padding=paddings.Valid) -> None:
        super().__init__()
        self.name = 'convolution'
        self.k = nr_kernels
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.stride = 1 #stride does not work with scipy conolve2d

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        kernel_shape = (self.k, self.prev_layer.o.shape[2], *self.kernel_size) # (Nr Kernels, input depth, x, y)

        self.w = np.random.uniform(-1.0, 1.0, kernel_shape)
        self.dw = np.zeros(self.w.shape)
        self.w_change = np.zeros(self.w.shape)
        
        self.b = np.random.uniform(-1.0, 1.0,(self.k,))
        self.db = np.zeros(self.b.shape)
        self.b_change = np.zeros(self.b.shape)

        self.process()
        self.summary = f'{self.name}\t{str(self.i.shape)}\t{str(self.o.shape)}\t{str(np.size(self.w) + np.size(self.b))}'
    
    def process(self):
        super().process()
        self.i_p = self.padding(self.i, self.kernel_size)
        
        # fft convolution # based on #https://laurentperrinet.github.io/sciblog/posts/2017-09-20-the-fastest-2d-convolution-in-the-world.html but with further performance improvements by myself.
        i_p_fft = fft2(self.i_p, axes=(0, 1))
        w_fft = np.moveaxis(fft2(self.w, s=self.i_p.shape[:2]), 1, -1)

        p = math.ceil(self.w.shape[2] / 2)
        self.net = np.zeros((self.i_p.shape[0] - p, self.i_p.shape[1] - p, self.w.shape[0]))
        for k in np.arange(self.w.shape[0]):
            self.net[:, :, k] = np.sum(np.real(ifft2(i_p_fft * w_fft[k], axes=(0, 1))), axis=2)[p:, p:] + self.b[k] # inverse transform and sum over c

        self.o = self.activation(self.net)

    def learn(self):
        super().learn()
        dy = self.activation(self.net, derivative=True) * self.dy

        if self.padding != paddings.Same:
            p = int((self.i.shape[0] - dy.shape[0]) / 2)
            dy_p = np.pad(dy, p)[:, :, p : -p]
        else:
            dy_p = dy

        # dw
        self.dw = np.zeros(self.w.shape)
        i_p_fft = np.moveaxis(fft2(self.i_p, axes=(0, 1)), -1, 0) # fft input and reshape to (c, h, w)
        dy_fft = np.resize(fft2(dy, s=self.i_p.shape[:2], axes=(0, 1)), (self.i_p.shape[-1], *self.i_p.shape[:2], dy.shape[-1])) # fft dy and reshape to (c, h, w, k)
        for k in np.arange(self.w.shape[0]):
            self.dw[k] = np.real(ifft2(i_p_fft * dy_fft[:, :, :, k]))[:, -self.w.shape[-2]:, -self.w.shape[-1]:] # multiply and inverse transform

        # dx
        w = np.flip(np.flip(self.w, axis=3), axis=2) # rotate filter
        w_fft = np.moveaxis(fft2(w, s=dy_p.shape[:2]), 0, -1) # fft filter and reshape to (c, h, w, k)
        dy_p_fft = np.resize(fft2(dy_p, axes=(0, 1)), (w.shape[1], *dy_p.shape)) # fft dy and reshape to (c, h, w, k)
        self.dx = np.moveaxis(np.sum(np.real(ifft2(dy_p_fft * w_fft, axes=(1, 2))), axis=-1), 0, -1)[-self.i.shape[0]:, -self.i.shape[1]:] # multiply, inverse transform, sum over kernels and reshape to (h, w, c)

        # db
        self.db = np.sum(self.dy, axis=(0, 1))


class Dropout(Layer):
    def __init__(self, drop_rate) -> None:
        super().__init__()
        self.name = 'dropout'
        self.drop_rate = drop_rate

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.process()
        self.summary = f'{self.name}\t\t{str(self.i.shape)}\t{str(self.o.shape)}\t0'
    
    def process(self):
        super().process()
        self.drop_map = np.random.choice([0, 1], self.i.shape, p=[self.drop_rate, 1 - self.drop_rate])
        drop = self.i * self.drop_map # set not relevant values to 0
        self.o = drop / (1 - self.drop_rate) # scale others https://stats.stackexchange.com/questions/219236/dropout-forward-prop-vs-back-prop-in-machine-learning-neural-network

    def learn(self):
        super().learn()
        drop = self.dy * self.drop_map
        self.dx = drop / (1 - self.drop_rate)