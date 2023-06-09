from model import utils, layers
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


class Network():

    def __init__(self, input_shape, layers) -> None:
        self.compiled = False
        self.history = []
        self.layers = []
        self.input_shape = input_shape

        if layers:
            for layer in layers:
                self.add_layer(layer)

    def add_layer(self, layer, input_layer = False) -> None:
        """Adds a layer object to the network model.
        
        Args:
            layer: layer object to be added.
            input_layer: Defines whether the layer is used as input to the neural network [optional].
        """
        if input_layer:
            self.layers.insert(0, layer)
        else:
            self.layers.append(layer)

            if layer.batch_norm is not None:
                self.layers.append(layer.activation)

            if layer.activation is not None:
                self.layers.append(layer.activation)

        self.compiled = False

    def compile(self, optimizer, loss, metric=None) -> None:
        """Compiles the model.
        
        Args:
            optimizer: Optimizer to be used to adjust weights and biases.
            loss: Loss function to be used to compute the loss value.
            metric: Metric functio to be used to evaluate the model [optional].
        """
        if not isinstance(self.layers[0], layers.Input):
            self.add_layer(layers.Input(self.input_shape), input_layer=True)
        if not isinstance(self.layers[-1], layers.Output):
            self.add_layer(layers.Output())

        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.compile(i, None, self.layers[i + 1])
            elif i == len(self.layers) - 1:
                layer.compile(i, self.layers[i - 1], None)
            else:
                layer.compile(i, self.layers[i - 1], self.layers[i + 1])

        self.optimizer = optimizer
        self.loss_function = loss
        self.metric = metric

        self.compiled = True

    def predict(self, input: np.ndarray) -> None:
        if input.ndim != 3:
            raise Exception('ShapeError: input shape must be of dim 3.')
        
        self.layers[0].x = input

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 4 or y.ndim != 2:
            raise Exception('Dimension must be 4 for input, 2 for output.')

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> None:
        """Evaluates the model using a defined metric function.

        Raises:
            FunctionError: If no function has been defined.
        """
        if self.metric is None:
            raise Exception('No metric defined.')
        
        self.metric(x, y, self, self.loss_function)

    def summary(self) -> None:
        """Gives an overview of the model architecture.
        
        Raises:
            Error: If the model has not been compiled yet.
        """
        if not self.compiled:
            raise Exception('Model has not been compiled yet.')
        
        print('%15s | %15s | %15s | %10s' % ('layer_type', 'input_shape', 'output_shape', 'parameters'))
        params = 0

        for l in self.layers [1:-1]:
            if l.is_activation_layer:
                continue

            ws = np.size(l.w) if l.w is not None else 0
            bs = np.size(l.b) if l.b is not None else 0
            params += ws + bs
            print('%15s | %15s | %15s | %10s' % (l.__class__.__name__, str(l.x.shape), str(l.y.shape), str(ws + bs)))

        print(f'\ntotal trainable parameters {params}')

    def plot_training_loss(self) -> None:
        """Plots the loss over epochs if the model has been trained yet. """
        plt.figure(figsize=(20,4))
        plt.plot(np.arange(len(self.history)), self.history)
        plt.xlabel('epoch')
        plt.ylabel('loss')

    def plot_neuron_activations(self) -> None:
        """Plots neuron activation distribution."""
        plt.figure(figsize=(20,4))
        legends = []
        for i, layer in enumerate(self.layers[1:-2]):
            if layer.is_activation_layer:
                print('layer %i (%s) | mean %.4f | std %.4f' % (i, layer.__class__.__name__, layer.y.mean(), layer.y.std()))
                X = layer.y.flatten()
                X.sort()
                density = gaussian_kde(X)
                density.covariance_factor = lambda : .25
                density._compute_covariance()
                plt.plot(X, density(X))
                legends.append('layer %i (%s)' % (i, layer.__class__.__name__))
        plt.legend(legends)
        plt.title('activation distribution')

    def plot_neuron_gradients(self) -> None:
        """Plots neuron gradient distribution."""
        plt.figure(figsize=(20,4))
        legends = []
        for i, layer in enumerate(self.layers[1:-2]):
            if layer.has_params:
                print('layer %i (%s) | mean %.4f | std %.4f' % (i, layer.__class__.__name__, layer.dw.mean(), layer.dw.std()))
                W = layer.dw.flatten()
                W.sort()
                density = gaussian_kde(W)
                density.covariance_factor = lambda : .25
                density._compute_covariance()
                plt.plot(W, density(W))
                legends.append('layer %i (%s)' % (i, layer.__class__.__name__))
        plt.legend(legends)
        plt.title('gradient distribution')


class FeedForward(Network):
    def __init__(self, input_shape, layers=[]) -> None:
        super().__init__(input_shape, layers)
    
    def __forward(self) -> None:
        for layer in self.layers:
            layer.forward()

    def predict(self, input: np.ndarray) -> np.ndarray:
        """Computes an output based on a single given input.
        
        Args:
            input: Array of input values.

        Returns:
            output: Array of output values.

        Raises:
            ShapeError: If input shape is not of dim 3.
        """
        super().predict(input)
        self.__forward()
        return self.layers[-1].y

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int=100, batch_size: int = None, log: bool=True) -> None:
        """Trains the model using given input data.
        
        Args:
            x: Array of input features.
            y: Array of training input.
            epochs: Number of epochs the training should last for [optional].
            batch_size: Number of input arrays used per epoch. If None, all training samples are used. [optional].
            log: If false, feedback per epoch is surpressed [optional].

        Raises:
            ShapeError: If feature array is not of dim 4 or training input array is not of dim 2. 
        """
        super().train(x, y)
        batch_size = batch_size if batch_size else len(x)
        loss_hist = []

        for epoch in range(1, epochs + 1):
            epoch_loss_hist = []
            x_shuffled, y_shuffled = utils.shuffle(x, y)
            start = time.time()

            for i, p in enumerate(x_shuffled[:batch_size]):
                if log:
                    print('epoch %5s/%5s | Training ... %i/%i' % (epoch, epochs, i + 1, batch_size), end='\r')

                loss, loss_gradient = self.loss_function(self.predict(p), np.squeeze(y_shuffled[i]))
                epoch_loss_hist.append(loss)
                self.optimizer.optimize(loss_gradient, self.layers)
            
            end = time.time()
            step = round((end - start) * 1000, 2)
            dim = 'ms'

            if step > 1000:
                step = round(step / 1000, 2)
                dim = 's'

            epoch_loss = sum(epoch_loss_hist) / len(epoch_loss_hist)
            loss_hist.append(epoch_loss)

            if log:   
                print('epoch %5s/%5s | time/epoch %.2f %s | loss %.4f' % (epoch, epochs, step, dim, round(epoch_loss, 4)))
        
        self.history = loss_hist