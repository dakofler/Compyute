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
    
    def learn(self):
        pass


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
        self.output = self.input
    
    def learn(self, learning_rate, momentum):
        super().learn()
        pass


class Output(Layers):
    def __init__(self) -> None:
        super().__init__()

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.process()

    def process(self):
        super().process()
        self.output = self.input
    
    def learn(self, loss):
        super().learn()
        self.learn_output = loss


class Flatten(Layers):
    def __init__(self) -> None:
        super().__init__()

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.process()

    def process(self):
        super().process()
        self.output = self.input.reshape((self.input.shape[0] ** 2, 1))
    
    def learn(self, learning_rate, momentum):
        super().learn()
        pass


class Dense(Layers):
    def __init__(self, nr_neurons, activation=activations.Identity) -> None:
        super().__init__()
        self.nr_neurons = nr_neurons
        self.activation = activation

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.weights = np.random.rand(self.nr_neurons, self.prev_layer.output.shape[0] + 1) * 2.0 - 1.0 # +1 for the bias neuron weights
        self.delta_weights = np.zeros(self.weights.shape)
        self.process()

    def process(self):
        super().process()
        input = np.append(self.input, [[1.0]], axis=0) # add bias neuron
        self.net = np.dot(self.weights, input)
        self.output = self.activation(self.net)
    
    def learn(self, learning_rate, momentum):
        super().learn()
        loss_gradient = self.succ_layer.learn_output # learn_output = t - o for output layer, sum (delta + w) for other layers

        if self.activation == activations.Softmax:
            d_softmax = self.activation(self.net, derivative=True)
            loss_gradient = np.reshape(loss_gradient, (1, -1))
            delta = (loss_gradient @ d_softmax).transpose()
        else:
            delta = self.activation(self.net, derivative=True) * loss_gradient

        w = self.weights.copy()
        w = np.delete(w, -1, axis=1) # remove weights to bias neuron
        self.learn_output = np.dot(w.transpose(), delta)
        self.delta_weights = learning_rate * np.append(self.prev_layer.output, [[1.0]], axis=0).transpose() * delta + momentum * self.delta_weights
        self.weights = self.weights + self.delta_weights


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
        self.output = np.ones((int(self.input.shape[0] / self.pooling_window[0]),  int(self.input.shape[1] / self.pooling_window[1])))
    
    def learn(self, learning_rate, momentum):
        super().learn()
        pass


class Convolutional(Layers):
    def __init__(self, nr_kernels, kernel_size) -> None:
        super().__init__()
        self.nr_kernels = nr_kernels
        self.kernel_size = kernel_size

    def integrate(self, id, prev_layer, succ_layer):
        super().integrate(id, prev_layer, succ_layer)
        self.process()
    
    def process(self):
        super().process()
        # ToDo, meanwhile
        self.output = np.ones((self.input.shape[0] - int((self.kernel_size[0] - 1) / 2) * 2,  self.input.shape[1] - int((self.kernel_size[1] - 1) / 2) * 2))

    def learn(self, learning_rate, momentum):
        super().learn()
        pass