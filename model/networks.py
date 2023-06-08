from model import utils
import numpy as np
import time
import matplotlib.pyplot as plt


class Network():
    def __init__(self, layers) -> None:
        self.compiled = False
        self.history = []
        self.layers = []
        if layers:
            for layer in layers:
                self.__add_layer(layer)

    def __add_layer(self, layer) -> None:
        self.layers.append(layer)

    def compile(self, optimizer, loss, metric) -> None:
        self.compiled = True
        self.optimizer = optimizer
        self.loss_function = loss
        self.metric = metric

    def propagate(self) -> None:
        pass

    def predict(self, input: np.ndarray) -> None:
        if input.ndim != 3: raise Exception('ShapeError: input shape must be of dim 3.')
        self.layers[0].i = input

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 4 or y.ndim != 2: raise Exception('Dimension must be 4 for input, 2 for output.')

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> None:
        """Evaluates the model using a defined metric function.

        Raises:
            FunctionError: If no function has been defined.
        """
        if self.metric is None: raise Exception('No metric defined.')
        loss, name, value, step = self.metric(x, y, self, self.loss_function)
        print (f'loss={round(loss, 4)}\t{name}={value}\ttime={step}ms')

    def summary(self) -> None:
        """Gives an overview of the model architecture.
        
        Raises:
            Error: If the model has not been compiled yet.
        """
        if not self.compiled: raise Exception('Model has not been compiled yet.')
        print(f'layer_type\tinput_shape\toutput_shape\tparameters')
        params = 0
        for l in self.layers:
            print(l.summary)
            if l.w is not None: params += np.size(l.w)
            if l.b is not None: params += np.size(l.b)
        print(f'total trainable parameters {params}')

    def plot_loss(self) -> None:
        """Plots the loss over epochs if the model has been trained yet.
        
        Raises:
            Error: If the model has not been trained yet.
        """
        if not self.history: raise Exception('Model has not been trained yet.')
        plt.plot(np.arange(len(self.history)), self.history)
        plt.xlabel('epoch')
        plt.ylabel('loss')


class FeedForward(Network):
    def __init__(self, layers=[]) -> None:
        super().__init__(layers)

    def compile(self, optimizer, loss, metric=None) -> None:
        """Compiles the model.
        
        Args:
            optimizer: Optimizer to be used to adjust weights and biases.
            loss: Loss function to be used to compute the loss value.
            metric: Metric functio to be used to evaluate the model [optional].
        """
        super().compile(optimizer, loss, metric)
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.integrate(i, None, self.layers[i + 1])
            elif i == len(self.layers) - 1:
                layer.integrate(i, self.layers[i - 1], None)
            else:
                layer.integrate(i, self.layers[i - 1], self.layers[i + 1])
    
    def __propagate(self) -> None:
        super().propagate()
        for layer in self.layers:
            layer.process()

    def predict(self, input: np.ndarray) -> np.ndarray:
        """Computes an output based on a given input.
        
        Args:
            input: Array of input values.

        Returns:
            output: Array of output values.

        Raises:
            ShapeError: If input shape is not of dim 3.
        """
        super().predict(input)
        self.__propagate()
        return self.layers[-1].o

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

            for i, p in enumerate(x_shuffled):
                if log: print(f'epoch {epoch}/{epochs}\tTraining ... {i + 1}/{batch_size}', end='\r')
                if i >= batch_size: break
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
                print (f'epoch {epoch}/{epochs}\ttime/epoch={step}{dim}\tloss={round(epoch_loss, 4)}')
        
        self.history = loss_hist