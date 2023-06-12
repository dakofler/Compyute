# neural networks module

from numpynn import layers, activations, preprocessing, utils
import numpy as np
import time
import matplotlib.pyplot as plt


class Network():
    """ Neural network base class """

    def __init__(self, input_shape, layers) -> None:
        self.compiled = False
        self.loss_history = []
        self.layers = []
        self.input_shape = input_shape

        if layers:
            for layer in layers:
                self.add_layer(layer)

    def __call__(self, x: np.ndarray) -> None:
        self.__check_dims(x)
        self.layers[0].x = x

    def compile(self, optimizer, loss_function, metric=None) -> None:
        """ Compiles the model.
        
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
        self.loss = loss_function
        self.metric = metric

        self.compiled = True

    def add_layer(self, layer, input_layer = False) -> None:
        """ Adds a layer object to the network model.
        
        Args:
            layer: layer object to be added.
            input_layer: Defines whether the layer is used as input to the neural network [optional].
        """
        
        if input_layer:
            self.layers.insert(0, layer)
        else:
            self.layers.append(layer)

            if layer.batch_norm is not None:
                self.layers.append(layer.batch_norm)

            if layer.activation is not None:
                self.layers.append(layer.activation)

        self.compiled = False


    def summary(self) -> None:
        """ Gives an overview of the model architecture.
        
        Raises:
            Error: If the model has not been compiled yet.
        """
        
        if not self.compiled:
            raise Exception('Model has not been compiled yet.')
        
        print('%15s | %15s | %15s | %15s | %15s | %10s\n' % ('layer_type', 'input_shape', 'weight_shape', 'bias_shape', 'output_shape', 'parameters'))
        params = 0

        for l in self.layers [1:-1]:
            if l.is_activation_layer:
                continue

            ws = np.size(l.w) if l.w is not None else 0
            w_shape = l.w.shape if l.w is not None else ()
            bs = np.size(l.b) if l.b is not None else 0
            b_shape = l.b.shape if l.b is not None else ()

            params += ws + bs
            print('%15s | %15s | %15s | %15s | %15s | %10s' % (l.__class__.__name__, str(l.x.shape[1:]), str(w_shape), str(b_shape), str(l.y.shape[1:]), str(ws + bs)))

        print(f'\ntotal trainable parameters {params}')

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        self.__check_dims(x, y)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> None:
        """ Evaluates the model using a defined metric function.

        Raises:
            FunctionError: If no function has been defined.
        """
        
        if self.metric is None:
            raise Exception('No metric defined.')
        
        self.metric(x, y, self)

    def plot_training_loss(self) -> None:
        """ Plots the loss over epochs if the model has been trained yet """

        plt.figure(figsize=(20,4))
        plt.plot(np.arange(len(self.loss_history)), self.loss_history)
        plt.xlabel('epoch')
        plt.ylabel('loss')

    def plot_neuron_activations(self, bins: int=100) -> None:
        """ Plots neuron activation distribution """

        plt.figure(figsize=(20,4))
        legends = []

        for i, layer in enumerate(self.layers[1:-1]):
            if layer.is_activation_layer and not isinstance(layer, activations.Softmax):
                print('layer %i (%s) | mean %.4f | std %.4f' % (i, layer.__class__.__name__, layer.y.mean(), layer.y.std()))
                
                Y, X = np.histogram(layer.y, bins=bins)
                X = np.delete(X, -1)
                plt.plot(X, Y)
                legends.append('layer %i (%s)' % (i, layer.__class__.__name__))

        plt.legend(legends)
        plt.title('activation distribution')

    def plot_neuron_gradients(self, bins: int=100) -> None:
        """ Plots neuron gradient distribution """

        plt.figure(figsize=(20,4))
        legends = []

        for i, layer in enumerate(self.layers[1:-2]):
            if layer.has_params:
                print('layer %i (%s) | mean %.4f | std %.4f' % (i, layer.__class__.__name__, layer.dw.mean(), layer.dw.std()))

                Y, X = np.histogram(layer.dw, bins=bins)
                X = np.delete(X, -1)
                plt.plot(X, Y)
                legends.append('layer %i (%s)' % (i, layer.__class__.__name__))

        plt.legend(legends)
        plt.title('gradient distribution')

    def plot_conv_activations(self) -> None:
        """ Plots neuron activations for convolutional layers """

        conv_layers = [l for l in self.layers if isinstance(l, layers.Convolution)]

        if not conv_layers:
            print('No convolutional layers found.')

        for i,l in enumerate(conv_layers):
            print(l.__class__.__name__, i + 1)
            plt.figure(figsize=(40, 40))

            for j in range(l.k):
                plt.subplot(10, 8, j + 1)
                plt.imshow(l.y[0, :, :, j], cmap='gray')
                plt.xlabel(f'kernel {str(j)}')

            plt.show()

    def __check_dims(self, x, y=None):
        req_input_dim = self.layers[0].x.ndim

        if x.ndim != req_input_dim:
            raise Exception(f'Isput dimension must be {req_input_dim}.')

        if y is not None:
            req_output_dim = self.layers[-1].y.ndim

            if y.ndim != req_output_dim:
                raise Exception(f'Output dimension must be {req_output_dim}.')

class FeedForward(Network):
    """Feed forward neural network model"""

    def __init__(self, input_shape, layers=[]) -> None:
        super().__init__(input_shape, layers)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Computes an output based on a single given input.
        
        Args:
            input: Array of input values.

        Returns:
            output: Array of output values.

        Raises:
            ShapeError: If input shape is not of dim 3.
        """
        
        super().__call__(x)
        self.__forward()
        return self.layers[-1].y

    def __forward(self) -> None:
        for layer in self.layers:
            layer.forward()

    def __backward(self) -> None:
        self.layers[-1].dy = self.loss.backward()  
        layers_reversed = self.layers.copy()
        layers_reversed.reverse()

        for layer in layers_reversed:
            layer.backward()

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
        self.loss_history = []

        for epoch in range(1, epochs + 1):
            start = time.time()
            x_shuffled, y_shuffled = utils.shuffle(x, y)
            x_train, y_train = x_shuffled[:batch_size], y_shuffled[:batch_size]

            loss = self.loss(self(x_train), y_train) # compute loss
            self.__backward() # backward pass
            self.optimizer(self) # update weights and biases

            end = time.time()
            step = round((end - start) * 1000, 2)

            if log:   
                print('epoch %5s/%5s | time/epoch %.2f ms | loss %.4f' % (epoch, epochs, step, loss))

            self.loss_history.append(loss)