from model import activations
import numpy as np

class Layers:
    def __init__(self):
        self.input = None
        self.output = None
        self.input_shape = None
        self.weights = None
        self.learn_output = None
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


class Input(Layers):
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
    

class Output(Layers):
    def __init__(self) -> None:
        super().__init__()

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.process()

    def process(self):
        super().process()
        self.output = self.input
    

class Flatten(Layers):
    def __init__(self) -> None:
        super().__init__()

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.process()

    def process(self):
        super().process()
        # self.output = self.input.reshape((self.input.shape[0] ** 2, 1))
        self.output = self.input.flatten()


class Dense(Layers):
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


class MaxPooling(Layers):
    def __init__(self, pooling_window) -> None:
        super().__init__()
        self.pooling_window = pooling_window

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.process()
    
    def process(self):
        super().process()
        # ToDo, meanwhile
        self.output = self.input
    

class Convolutional(Layers):
    def __init__(self, nr_kernels, kernel_size=(3, 3), activation=activations.Identity) -> None:
        super().__init__()
        self.nr_kernels = nr_kernels
        self.kernel_size = kernel_size
        self.kernels = []
        for i in range(nr_kernels):
            self.kernels.append(np.random.uniform(-1.0, 1.0, self.kernel_size))
        self.activation = activation

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.process()
    
    def process(self):
        super().process()
        
        # https://stackoverflow.com/questions/19414673/in-numpy-how-to-efficiently-list-all-fixed-size-submatrices
        shape = self.kernels[0].shape + tuple(np.subtract(self.input.shape, self.kernels[0].shape) + 1)
        strd = np.lib.stride_tricks.as_strided
        sub_images = strd(self.input, shape = shape, strides = self.input.strides * 2)
        self.output = np.zeros((self.nr_kernels, shape[2], shape[3]))

        for i, k in enumerate(self.kernels):
            self.output[i] = self.activation(np.einsum('ij,ijkl->kl', k, sub_images))