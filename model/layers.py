import numpy as np

class Layers:
    def __init__(self):
        self.input = None

    def integrate(self, id, prev_layer):
        self.id = id
        self.prev_layer = prev_layer

    def process(self):
        pass


class Flatten(Layers):
    def __init__(self, input_shape=None) -> None:
        super().__init__()
        self.input_shape = input_shape

    def integrate(self, id, prev_layer=None):
        super().integrate(id, prev_layer)
        if self.prev_layer is not None:
            self.input_shape = self.prev_layer.output.shape
        self.output = np.ones((self.input_shape[0] ** 2, 1))

    def process(self):
        super().process()
        if self.prev_layer is not None:
            self.output = self.prev_layer.output.reshape(self.output.shape)
        else:
            self.input.reshape(self.output.shape)
        

class Dense(Layers):
    def __init__(self, nr_neurons, activation) -> None:
        super().__init__()
        self.nr_neurons = nr_neurons
        self.activation = activation

    def integrate(self, id, prev_layer=None):
        super().integrate(id, prev_layer)

        if self.prev_layer is not None:
            self.input_shape = self.prev_layer.output.shape
            self.weights = np.random.rand(self.nr_neurons, self.prev_layer.output.shape[0] + 1) * 2.0 - 1.0 # +1 for the bias neuron weights
        else: self.input_shape = (self.nr_neurons, 1)

        self.output = np.ones((self.nr_neurons, 1))

    def process(self):
        super().process()
        if self.prev_layer is not None:
            input = np.append(self.prev_layer.output, [[1.0]], axis=0) # add bias neuron
            self.net = np.dot(self.weights, input)
            self.output = self.activation(self.net)
        else:
            self.output = self.input



class MaxPooling(Layers):
    def __init__(self, pooling_window) -> None:
        super().__init__()
        self.pooling_window = pooling_window

    def integrate(self, id, prev_layer=None):
        super().integrate(id, prev_layer)


class Convolutional(Layers):
    def __init__(self, nr_kernels, kernel_size, input_shape=None) -> None:
        super().__init__()
        self.nr_kernels = nr_kernels
        self.kernel_size = kernel_size
        self.input_shape = input_shape

    def integrate(self, id, prev_layer=None):
        super().integrate(id, prev_layer)

        if self.prev_layer is not None:
            self.input_shape = self.prev_layer.output.shape
        self.output = np.ones(self.input_shape)
    
    def process(self):
        super().process()
        if self.prev_layer is not None:
            self.input = self.prev_layer.output
        
        self.output = self.input