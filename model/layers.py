from model import activations, paddings
import numpy as np
import math

class Layer:
    def __init__(self):
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
        self.input_shape = input_shape

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.i = np.ones(self.input_shape)
        self.process()

    def process(self):
        super().process()
        self.o = self.i.reshape(self.input_shape)
    
    def learn(self):
        super().learn()
    

class Output(Layer):
    def __init__(self) -> None:
        super().__init__()

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.process()

    def process(self):
        super().process()
        self.o = self.i
    
    def learn(self):
        super().learn()


class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.process()

    def process(self):
        super().process()
        self.o = self.i.flatten()
    
    def learn(self):
        super().learn()
        self.dx = self.dy.reshape(self.i.shape)


class Dense(Layer):
    def __init__(self, nr_neurons, activation=activations.Identity) -> None:
        super().__init__()
        self.nr_neurons = nr_neurons
        self.activation = activation

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.w = np.random.uniform(-1.0, 1.0, (self.nr_neurons, self.prev_layer.o.shape[0] + 1)) # +1 for the bias neuron weights
        self.dw = np.zeros(self.w.shape)
        self.w_change = np.zeros(self.w.shape)
        self.process()

    def process(self):
        super().process()
        input = np.append(self.i, [1.0], axis=0) # add bias neuron
        self.net = np.dot(self.w, input)
        self.o = self.activation(self.net)

    def learn(self):
        super().learn()

        # derivatie of activation function
        if self.activation == activations.Softmax: # https://e2eml.school/softmax.html
            d_softmax = self.activation(self.net, derivative=True)
            self.dy = np.reshape(self.dy, (1, -1))
            dy = np.squeeze(self.dy @ d_softmax)
        else:
            dy = self.activation(self.net, derivative=True) * self.dy

        w = np.delete(self.w.copy(), -1, axis=1) # remove weights corresponding to bias neurons
        self.dx = np.dot(w.T, dy) # compute loss gradient to be used in the previous layer before weights are changed
        self.dw = np.append(self.prev_layer.o, [1.0], axis=0) * np.expand_dims(dy, 1)


class MaxPooling(Layer):
    def __init__(self, pooling_window) -> None:
        super().__init__()
        self.pooling_window = pooling_window

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.process()
    
    def process(self):
        super().process()
        input_height, input_width, input_depth = self.i.shape
        step = self.pooling_window[0]
        output_width = int(input_width / step)
        output_height = int(input_height / step)

        self.o = np.zeros((output_height, output_width, input_depth))
        self.dx_map = np.zeros((input_height, input_width, input_depth))

        for f in range(input_depth):
            image = self.i[:, :, f]
            for y in range(output_height):
                for x in range(output_width):
                    a = image[y * step : y * step + step, x * step : x * step + step] # get sub matrix
                    index = np.where(a == np.max(a)) # get index of max value in sub matrix
                    index_y = index[0][0] + y * step # get y index of max value in input matrix
                    index_x = index[1][0] + x * step # get x index of max value in input matrix

                    self.o[y, x, f] = image[index_y, index_x]
                    self.dx_map[index_y, index_x, f] = 1

    def learn(self):
        super().learn()
        dy = np.repeat(self.dy, self.pooling_window[0], axis=0)
        dy = np.repeat(dy, self.pooling_window[0], axis=1)
        dy = np.resize(dy, self.dx_map.shape) # if input mod pooling size is not 0, gradient and map shape is not equal. Resize fills missing values with 0.
        self.dx = dy * self.dx_map


class Convolution(Layer):
    def __init__(self, nr_kernels, kernel_size=(3, 3), activation=activations.Identity, padding=paddings.Same, stride=1) -> None:
        super().__init__()
        self.k = nr_kernels
        self.kernel_size = kernel_size
        self.padding_size = math.floor(self.kernel_size[0] / 2)
        self.activation = activation
        self.padding = padding
        self.stride = stride

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        kernel_shape = (self.k, *self.kernel_size, self.prev_layer.o.shape[2]) # (Nr Kernels, x, y, colorchannels)

        self.w = np.random.uniform(-1.0, 1.0, kernel_shape)
        self.dw = np.zeros(self.w.shape)
        self.w_change = np.zeros(self.w.shape)
        
        self.b = np.random.uniform(-1.0, 1.0,(self.k,))
        self.db = np.zeros(self.b.shape)
        self.b_change = np.zeros(self.b.shape)

        self.process()
    
    def process(self):
        super().process()
        # output_size = int((self.i.shape[0] - 2 * (self.padding == paddings.Same) * self.padding_size) / self.stride)
        i_p = self.padding(self.i, self.padding_size)
        output_size = int((i_p.shape[0] - self.kernel_size[0]) / self.stride) + 1
        self.net = np.zeros((output_size, output_size, self.k))
        
        # convolution
        for k, kernel in enumerate(self.w):
            y_count = 0
            for y in range(0, output_size * self.stride, self.stride):
                x_count = 0
                for x in range(0, output_size * self.stride, self.stride):
                    array = i_p[y : y + self.kernel_size[0], x : x + self.kernel_size[1], :]
                    self.net[y_count, x_count, k] = np.sum(array * kernel) + self.b[k]
                    x_count += 1
                y_count += 1

        self.o = self.activation(self.net)

    def learn(self):
        super().learn()
        i_p = self.padding(self.i, self.padding_size)
        dw_size = int((i_p.shape[0] - self.dy.shape[0]) / self.stride) + 1
        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros(self.b.shape)
        self.dx = np.zeros(self.i.shape)

        dy = self.activation(self.net, derivative=True) * self.dy
        dy_p = self.padding(dy, self.padding_size)
        dx_size = int((dy_p.shape[0] - self.kernel_size[0]) / self.stride) + 1

        for k, kernel in enumerate(self.w):
            for c in range(i_p.shape[2]):
                #dw
                y_count = 0
                for y in range(0, dw_size * self.stride, self.stride):
                    x_count = 0
                    for x in range(0, dw_size * self.stride, self.stride):
                        array = i_p[y : y + dy.shape[0], x : x + dy.shape[0], c]
                        self.dw[k, y_count, x_count, c] += np.sum(array * dy[:, :, k])
                        x_count += 1
                    y_count += 1
                
                #dx
                y_count = 0
                for y in range(0, dx_size * self.stride, self.stride):
                    x_count = 0
                    for x in range(0, dx_size * self.stride, self.stride):
                        array = dy_p[y : y + self.kernel_size[0], x : x + self.kernel_size[1], k]
                        self.dx[y_count, x_count, c] += np.sum(array * np.flipud(np.fliplr(kernel)))
                        x_count += 1
                    y_count += 1
            
            #db
            self.db[k] = np.sum(self.dy[:, :, k])