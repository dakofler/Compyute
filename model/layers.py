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
        input_feature_map = self.input.shape[2]
        step = self.pooling_window[0]
        output_width = int(input_width / step)
        output_height = int(input_height / step)

        self.output = np.zeros((output_height, output_width, input_feature_map))

        for f in range(input_feature_map):
            image = self.input[:, :, f]
            for y in range(output_height):
                for x in range(output_width):
                    arr = image[y * step : y * step + step, x * step : x * step + step]
                    self.output[y, x, f] = np.max(arr)

    def learn(self):
        super().learn()
        pass


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
            sobel = np.array([
                [[1], [0], [-1]],
                [[2], [0], [-2]],
                [[1], [0], [-1]]])
            ext = np.ones((1, 1, kernel_shape[2]))
            self.kernels.append(sobel * ext)
        else:
            for i in range(self.nr_kernels):
                self.kernels.append(np.random.uniform(-1.0, 1.0, kernel_shape))

        self.process()
    
    def process(self):
        super().process()
        # https://medium.com/@thepyprogrammer/2d-image-convolution-with-numpy-with-a-handmade-sliding-window-view-946c4acb98b4

        # self coded convolution
        # https://dev.to/sandeepbalachandran/machine-learning-convolution-with-color-images-2p41      
        kernel_overhang = int((self.kernels[0].shape[0] - 1) / 2)
        test_image = self.input[:, :, 0]
        test_image_p = self.padding(test_image)
        feature_map_height = test_image_p.shape[0] - 2 * kernel_overhang
        feature_map_width = test_image_p.shape[1] - 2 * kernel_overhang

        feature_map = np.zeros((int(feature_map_height / self.stride), int(feature_map_width / self.stride), self.nr_kernels, self.input.shape[2]))
        for k, kernel in enumerate(self.kernels): # ToDo: do all channels at once
            for channel in range(self.input.shape[2]):
                image = self.input[:, :, channel]
                filter = kernel[:, :, channel]

                image_p = self.padding(image) # padding ToDo: Multichannel image padding

                width = image_p.shape[1]
                height = image_p.shape [0]
                filter_size = filter.shape[0]

                # convolution
                for y in range(int((height - 2 * kernel_overhang) / self.stride)):
                    for x in range(int((width - 2 * kernel_overhang) / self.stride)):
                        arr = image_p[y * self.stride :  y * self.stride + filter_size, x * self.stride : x * self.stride + filter_size]
                        feature_map[y, x, k, channel] = np.sum(arr * filter)
        
        feature_maps = np.sum(feature_map, axis=3) # sum over channels
        feature_maps = feature_maps + self.bias.reshape((1, 1, len(self.bias))) # add bias to each feature map
        self.output = self.activation(feature_maps)

    def learn(self):
        super().learn()
        pass