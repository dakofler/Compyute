import numpy as np

class Layers:
    def __init__(self):
        pass

    def integrate(self, id, prev_layer):
        self.id = id
        self.prev_layer = prev_layer

    def process(self):
        pass


class Flatten(Layers):
    def __init__(self) -> None:
        super().__init__()

    def integrate(self, id, prev_layer):
        self.id = id
        self.prev_layer = prev_layer
        input_shape = self.prev_layer.output.shape
        output_shape = (input_shape[0] ** 2, 1)

        self.output = np.zeros(output_shape)

    def process(self):
        self.output = self.prev_layer.output.reshape(self.output.shape)
        

class Dense(Layers):
    def __init__(self, nr_neurons, activation) -> None:
        super().__init__()
        self.nr_neurons = nr_neurons
        self.activation = activation

    def integrate(self, id, prev_layer):
        self.id = id
        self.prev_layer = prev_layer

        if self.prev_layer is not None:
            self.weights = np.random.rand(self.nr_neurons, self.prev_layer.output.shape[0] + 1) * 2.0 - 1.0 # +1 for the bias neuron weights
        self.output = np.zeros((self.nr_neurons, 1))

    def process(self):
        input = np.append(self.prev_layer.output, [[1.0]], axis=0) # add bias neuron
        self.net = np.dot(self.weights, input)
        self.output = self.activation(self.net)


class MaxPooling(Layers):
    def __init__(self, pooling_window) -> None:
        super().__init__()

    def integrate(self, id, prev_layer):
        self.id = id
        self.prev_layer = prev_layer


class Convolutional(Layers):
    def __init__(self, input_shape, kernel_size) -> None:
        super().__init__()
        self.input_shape = input_shape

        # for testing
        self.output = np.ones(self.input_shape)

    def integrate(self, id, prev_layer=None):
        self.id = id
        self.prev_layer = prev_layer
    
    def process(self):
        if self.prev_layer is not None:
            self.input = self.prev_layer.output
        
        self.output = self.input