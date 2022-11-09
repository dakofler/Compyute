from model import activations, paddings
import numpy as np


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
        pass
    

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
        pass


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

        # https://e2eml.school/softmax.html
        if self.activation == activations.Softmax:
            d_softmax = self.activation(self.net, derivative=True)
            self.succ_loss_gradient = np.reshape(self.succ_loss_gradient, (1, -1))
            d = np.squeeze(self.succ_loss_gradient @ d_softmax)
        else:
            d = self.activation(self.net, derivative=True) * self.succ_loss_gradient

        w = np.delete(self.weights.copy(), -1, axis=1) # remove weights corresponding to bias neurons
        self.loss_gradient = np.dot(w.transpose(), d) # compute loss gradient to be used in the next layer before weights are changed
        return np.append(self.prev_layer.output, [1.0], axis=0) * np.expand_dims(d, 1)


class MaxPooling(Layer):
    def __init__(self, pooling_window) -> None:
        super().__init__()
        self.pooling_window = pooling_window

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.process()
    
    def process(self):
        super().process()
        input_height = self.input.shape[0]
        input_width = self.input.shape[1]
        input_depth = self.input.shape[2]
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
        succ_loss_gradient.resize(self.loss_gradient_map.shape) # if input mod pooling size is not 0, gradient and map shape is not equal. Resize fills missing values with 0.
        self.loss_gradient = succ_loss_gradient * self.loss_gradient_map   


class Convolution(Layer):
    def __init__(self, nr_kernels, kernel_size=(3, 3), activation=activations.Identity, padding=paddings.Same, stride=1) -> None:
        super().__init__()
        self.nr_kernels = nr_kernels
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.stride = stride

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.bias = np.ones((self.nr_kernels,))
        self.kernels = []
        kernel_shape = (*self.kernel_size, self.prev_layer.output.shape[2])

        if self.nr_kernels == 1:
            kernel = np.array([
                [[1], [0], [-1]],
                [[2], [0], [-2]],
                [[1], [0], [-1]]]) * np.ones((1, 1, kernel_shape[2]))
            self.kernels.append(kernel)
        else:
            for i in range(self.nr_kernels):
                self.kernels.append(np.random.uniform(-1.0, 1.0, kernel_shape))

        self.process()
    
    def process(self):
        super().process()
        # faster convolution https://medium.com/@thepyprogrammer/2d-image-convolution-with-numpy-with-a-handmade-sliding-window-view-946c4acb98b4
        # self coded convolution https://dev.to/sandeepbalachandran/machine-learning-convolution-with-color-images-2p41   
        kernel_overhang = int((self.kernels[0].shape[0] - 1) / 2)
        if self.padding == paddings.Same:
            featuremap_shape = (int((self.input.shape[0] - 2 * kernel_overhang) / self.stride), int((self.input.shape[1] - 2 * kernel_overhang) / self.stride), self.nr_kernels)
        else:
            featuremap_shape = (int(self.input.shape[0] / self.stride), int(self.input.shape[1] / self.stride), self.nr_kernels)
        feature_map = np.zeros(featuremap_shape)

        for k, kernel in enumerate(self.kernels):
            input = self.padding(self.input, kernel_overhang)

            # convolution
            for y in range(int((input.shape[0] - 2 * kernel_overhang) / self.stride)):
                for x in range(int((input.shape[1] - 2 * kernel_overhang) / self.stride)):
                    arr = input[y * self.stride :  y * self.stride + kernel.shape[0], x * self.stride : x * self.stride + kernel.shape[1], :]
                    feature_map[y, x, k] = np.sum(arr * kernel) + self.bias[k]

        self.output = self.activation(feature_map)

    def learn(self):
        super().learn()
        self.loss_gradient = self.succ_loss_gradient