from model import activations, paddings
import numpy as np
import math

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.input_shape = None
        self.weights = None
        self.loss_gradient = None
        self.pooling_window = None
        self.nr_kernels = None
        self.kernel_size = None
        self.delta = None
        self.bias_delta = None

    def integrate(self, id, prev_layer, succ_layer):
        self.id = id
        self.prev_layer = prev_layer
        self.succ_layer = succ_layer

    def process(self):
        if self.prev_layer is not None:
            self.input = self.prev_layer.output
    
    def learn(self):
        if self.succ_layer is not None:
            self.succ_loss_gradient = self.succ_layer.loss_gradient


class Input(Layer):
    def __init__(self, input_shape) -> None:
        super().__init__()
        self.input_shape = input_shape

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.input = np.ones(self.input_shape)
        self.process()

    def process(self):
        super().process()
        self.output = self.input.reshape(self.input_shape)
    
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
        self.output = self.input
    
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
        self.output = self.input.flatten()
    
    def learn(self):
        super().learn()
        self.loss_gradient = self.succ_loss_gradient.reshape(self.input.shape)


class Dense(Layer):
    def __init__(self, nr_neurons, activation=activations.Identity) -> None:
        super().__init__()
        self.nr_neurons = nr_neurons
        self.activation = activation

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.weights = np.random.uniform(-1.0, 1.0, (self.nr_neurons, self.prev_layer.output.shape[0] + 1)) # +1 for the bias neuron weights
        self.delta_weights = np.zeros(self.weights.shape)
        self.process()

    def process(self):
        super().process()
        
        input = np.append(self.input, [1.0], axis=0) # add bias neuron
        self.net = np.dot(self.weights, input)
        self.output = self.activation(self.net)

    def learn(self):
        super().learn()

        # derivatie of activation function
        if self.activation == activations.Softmax: # https://e2eml.school/softmax.html
            d_softmax = self.activation(self.net, derivative=True)
            self.succ_loss_gradient = np.reshape(self.succ_loss_gradient, (1, -1))
            d = np.squeeze(self.succ_loss_gradient @ d_softmax)
        else:
            d = self.activation(self.net, derivative=True) * self.succ_loss_gradient

        w = np.delete(self.weights.copy(), -1, axis=1) # remove weights corresponding to bias neurons
        self.loss_gradient = np.dot(w.transpose(), d) # compute loss gradient to be used in the previous layer before weights are changed
        self.delta = np.append(self.prev_layer.output, [1.0], axis=0) * np.expand_dims(d, 1)


class MaxPooling(Layer):
    def __init__(self, pooling_window) -> None:
        super().__init__()
        self.pooling_window = pooling_window

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.process()
    
    def process(self):
        super().process()
        input_height, input_width, input_depth = self.input.shape
        step = self.pooling_window[0]
        output_width = int(input_width / step)
        output_height = int(input_height / step)

        self.output = np.zeros((output_height, output_width, input_depth))
        self.loss_gradient_map = np.zeros((input_height, input_width, input_depth))

        for f in range(input_depth):
            image = self.input[:, :, f]
            for y in range(output_height):
                for x in range(output_width):
                    arr = image[y * step : y * step + step, x * step : x * step + step] # get sub matrix
                    index = np.where(arr == np.max(arr)) # get index of max value in sub matrix
                    index_y = index[0][0] + y * step # get y index of max value in input matrix
                    index_x = index[1][0] + x * step # get x index of max value in input matrix

                    self.output[y, x, f] = image[index_y, index_x]
                    self.loss_gradient_map[index_y, index_x, f] = 1

    def learn(self):
        super().learn()
        succ_loss_gradient = np.repeat(self.succ_loss_gradient, self.pooling_window[0], axis=0)
        succ_loss_gradient = np.repeat(succ_loss_gradient, self.pooling_window[0], axis=1)
        succ_loss_gradient = np.resize(succ_loss_gradient, self.loss_gradient_map.shape) # if input mod pooling size is not 0, gradient and map shape is not equal. Resize fills missing values with 0.
        self.loss_gradient = succ_loss_gradient * self.loss_gradient_map


class Convolution(Layer):
    def __init__(self, nr_kernels, kernel_size=(3, 3), activation=activations.Identity, padding=paddings.Same, stride=1) -> None:
        super().__init__()
        self.nr_kernels = nr_kernels
        self.kernel_size = kernel_size
        self.padding_size = math.floor(self.kernel_size[0] / 2)
        self.activation = activation
        self.padding = padding
        self.stride = stride

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        kernel_shape = (self.nr_kernels, *self.kernel_size, self.prev_layer.output.shape[2]) # (Nr Kernels, x, y, colorchannels)

        self.weights = np.random.uniform(-1.0, 1.0, kernel_shape)
        self.delta_weights = np.zeros(self.weights.shape)
        
        self.biases = np.random.uniform(-1.0, 1.0,(self.nr_kernels,))
        self.delta_biases = np.zeros(self.biases.shape)

        self.process()
    
    def process(self):
        super().process()
        output_size = int((self.input.shape[0] - 2 * (self.padding == paddings.Same) * self.padding_size) / self.stride)
        self.net = np.zeros((output_size, output_size, self.nr_kernels))
        input = self.padding(self.input, self.padding_size)

        for k, kernel in enumerate(self.weights):
            # convolution
            y_count = 0
            for y in range(0, input.shape[0] - int(self.kernel_size[0] / 2) - 1, self.stride):
                x_count = 0
                for x in range(0, input.shape[0] - int(self.kernel_size[1] / 2) - 1, self.stride):
                    array = input[y : y + self.kernel_size[0], x : x + self.kernel_size[1], :]
                    self.net[y_count, x_count, k] = np.sum(array * kernel) + self.biases[k]
                    x_count += 1
                y_count += 1

        # possible alternative fft -> product -> ifft = O(n logn)

        self.output = self.activation(self.net)

    def learn(self):
        super().learn()

        self.delta = np.zeros(self.delta_weights.shape)
        self.bias_delta = np.zeros(self.delta_biases.shape)
        self.loss_gradient = np.zeros(self.input.shape)
        input = self.padding(self.input, self.padding_size)
        succ_loss_gradient = self.activation(self.succ_loss_gradient, derivative=True)

        #dw = x * dy
        for k, kernel in enumerate(self.weights):
            for c in range(input.shape[2]):
                y_count = 0
                for y in range(0, input.shape[0] - int(self.kernel_size[0] / 2) - 1, self.stride):
                    x_count = 0
                    for x in range(0, input.shape[0] - int(self.kernel_size[1] / 2) - 1, self.stride):
                        array = input[y : y + self.kernel_size[0], x : x + self.kernel_size[1], c]
                        self.delta[k, y_count, x_count, c] = np.sum(array * succ_loss_gradient[:, :, k])
                        x_count += 1
                    y_count += 1
        
        # ToDo
        #dx
        # if (succ_loss_gradient.shape != 

        for k, kernel in enumerate(self.weights):
            y_count = 0
            for y in range(0, input.shape[0] - int(self.kernel_size[0] / 2) - 1, self.stride):
                x_count = 0
                for x in range(0, input.shape[0] - int(self.kernel_size[1] / 2) - 1, self.stride):
                    array = input[y : y + self.kernel_size[0], x : x + self.kernel_size[1], :]

                    
                    self.delta[k] += array * self.succ_loss_gradient[y_count, x_count, k]
                    self.loss_gradient[y : y + self.kernel_size[0], x : x + self.kernel_size[1], :] += self.succ_loss_gradient[y_count, x_count, k] * kernel
                    x_count += 1
                y_count += 1

        #db
        for k, kernel in enumerate(self.weights):
            self.bias_delta[k] = np.sum(self.succ_loss_gradient[:, :, k])
        

        # for k, kernel in enumerate(self.weights):
        #     y_count = 0
        #     for y in range(0, input.shape[0] - int(self.kernel_size[0] / 2) - 1, self.stride):
        #         x_count = 0
        #         for x in range(0, input.shape[0] - int(self.kernel_size[1] / 2) - 1, self.stride):
        #             array = input[y : y + self.kernel_size[0], x : x + self.kernel_size[1], :]

                    
        #             self.delta[k] += array * self.succ_loss_gradient[y_count, x_count, k]
        #             self.loss_gradient[y : y + self.kernel_size[0], x : x + self.kernel_size[1], :] += self.succ_loss_gradient[y_count, x_count, k] * kernel
        #             x_count += 1
        #         y_count += 1