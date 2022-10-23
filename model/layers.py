import numpy as np

class Layers:
    def __init__(self):
        self.input = None

    def integrate(self, id, prev_layer):
        self.id = id
        self.prev_layer = prev_layer

    def process(self):
        if self.prev_layer is not None:
            self.input = self.prev_layer.output


class Input(Layers):
    def __init__(self, input_shape) -> None:
        super().__init__()
        self.input_shape = input_shape

    def integrate(self, id):
        super().integrate(id, None)
        self.output = np.ones(self.input_shape)

    def process(self):
        super().process()
        self.output = self.input


class Flatten(Layers):
    def __init__(self) -> None:
        super().__init__()

    def integrate(self, id, prev_layer):
        super().integrate(id, prev_layer)
        self.process()

    def process(self):
        super().process()
        self.output = self.input.reshape((self.input.shape[0] ** 2, 1))
        

class Dense(Layers):
    def __init__(self, nr_neurons, activation) -> None:
        super().__init__()
        self.nr_neurons = nr_neurons
        self.activation = activation

    def integrate(self, id, prev_layer):
        super().integrate(id, prev_layer)
        self.weights = np.random.rand(self.nr_neurons, self.prev_layer.output.shape[0] + 1) * 2.0 - 1.0 # +1 for the bias neuron weights
        self.process()

    def process(self):
        super().process()
        input = np.append(self.input, [[1.0]], axis=0) # add bias neuron
        self.net = np.dot(self.weights, input)
        self.output = self.activation(self.net)


class MaxPooling(Layers):
    def __init__(self, pooling_window) -> None:
        super().__init__()
        self.pooling_window = pooling_window

    def integrate(self, id, prev_layer):
        super().integrate(id, prev_layer)
        self.process()
    
    def process(self):
        super().process()
        # ToDo, meanwhile
        self.output = np.ones((int(self.input.shape[0] / self.pooling_window[0]),  int(self.input.shape[1] / self.pooling_window[1])))


class Convolutional(Layers):
    def __init__(self, nr_kernels, kernel_size) -> None:
        super().__init__()
        self.nr_kernels = nr_kernels
        self.kernel_size = kernel_size

    def integrate(self, id, prev_layer):
        super().integrate(id, prev_layer)
        self.process()
    
    def process(self):
        super().process()
        # ToDo, meanwhile
        self.output = np.ones((self.input.shape[0] - int((self.kernel_size[0] - 1) / 2) * 2,  self.input.shape[1] - int((self.kernel_size[1] - 1) / 2) * 2))