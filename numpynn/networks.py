# neural network module

from numpynn import layers, activations, utils, optimizers, losses
import numpy as np
import matplotlib.pyplot as plt
import time


class Sequential():
    """Feed forward neural network model"""

    def __init__(self, input_shape: tuple[int], layers: list[layers.Layer]) -> None:
        self.input_shape = input_shape
        self.layers = []

        for layer in layers:
            self.__add_layer(layer)

        self.compiled = False

    def __call__(self, x: np.ndarray, y: np.ndarray=None) -> None:
        """Computes an output and the loss based on imput samples.
        
        Args:
            x: Array of input values.
            y: Array of target values [optional].

        Returns:
            output: Array of output values.
            loss: loss value, if target values are provided, else None.

        Raises:
            ShapeError: If input shape is not of dim 3.
        """
        self.__check_dims(x, y)
        self.layers[0].x = x.astype('float32')
        self.__forward()
        output = self.layers[-1].y
        loss = None

        if y is not None:
            loss = self.loss_function(output, y)
        return output, loss

    def compile(self, optimizer: optimizers.Optimizer, loss_function: losses.Loss, metric) -> None:
        """ Compiles the model.
        
        Args:
            optimizer: Optimizer to be used to update weights and biases.
            loss: Loss function to be used to compute the loss value and gradients.
            metric: Metric function to be used to evaluate the model.
        """
        if not isinstance(self.layers[0], layers.Input):
            self.__add_layer(layers.Input(self.input_shape), input_layer=True)

        if not isinstance(self.layers[-1], layers.Output):
            self.__add_layer(layers.Output())

        for i, l in enumerate(self.layers):
            if isinstance(l, layers.Input):
                l.compile(i, None, self.layers[i + 1])
            elif isinstance(l, layers.Output):
                l.compile(i, self.layers[i - 1], None)
            else:
                l.compile(i, self.layers[i - 1], self.layers[i + 1])

        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metric = metric
        self.compiled = True

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int=100, batch_size: int = None,
              verbose: bool=True, val_data: tuple[np.ndarray]=(None, None)) -> list[float]:
        """Trains the model using samples and targets.
        
        Args:
            x: Array of input features.
            y: Array of training input.
            epochs: Number of epochs the training should last for. [optional]
            batch_size: Number of input arrays used per epoch. If None, all training samples are used. [optional]
            verbose: If false, feedback per epoch is supressed. [optional]
            val_data: Data used for validation during training. [optional]

        Returns:
            losses: Loss values per epoch.

        Raises:
            ShapeError: If feature array is not of dim 4 or training input array is not of dim 2. 
        """
        self.__check_dims(x, y)
        history = []
        val_loss = None

        for epoch in range(1, epochs + 1):
            start = time.time()
            x_train, y_train = utils.shuffle(x, y, batch_size)

            # training
            self.__train()
            _, loss = self(x_train, y_train)
            self.__backward()
            self.optimizer(self)

            # validation
            self.__eval()

            if val_data[0] is not None:
                _, val_loss = self(*val_data)

            end = time.time()
            step = round((end - start) * 1000.0, 2)
            self.__log(epoch, epochs, step, loss, verbose, val_loss)
            history.append(loss)
        
        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Applies the input to the model and returns it's predictions.
        
        Args:
            x: Array of input features.

        Returns:
            output: Array of output values.
        """
        self.__eval()
        pred, _ = self(x)
        return pred

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> None:
        """ Evaluates the model using a defined metric function.

        Raises:
            FunctionError: If no function has been defined.
        """
        if self.metric is None:
            raise Exception('No metric defined.')
        
        self.__eval()
        self.metric(x, y, self)

    def summary(self) -> None:
        """ Gives an overview of the model architecture.
        
        Raises:
            Error: If the model has not been compiled yet.
        """
        if not self.compiled:
            raise Exception('Model has not been compiled yet.')
        
        print('%15s | %20s | %20s | %20s | %20s | %10s\n' % ('layer_type', 'input_shape', 'weight_shape', 'bias_shape', 'output_shape', 'parameters'))
        params = 0

        for l in self.layers [1:-1]:
            if isinstance(l, layers.Input) or isinstance(l, layers.Output):
                continue

            ws = np.size(l.w) if l.w is not None else 0
            w_shape = l.w.shape if l.w is not None else ()
            bs = np.size(l.b) if l.b is not None else 0
            b_shape = l.b.shape if l.b is not None else ()
            params += (ws + bs)
            print('%15s | %20s | %20s | %20s | %20s | %10s' % (l.__class__.__name__, str(l.x.shape[1:]), str(w_shape), str(b_shape), str(l.y.shape[1:]), str(ws + bs)))

        print(f'\ntotal trainable parameters {params}')

    def plot_training_loss(self, history) -> None:
        """ Plots the loss over epochs if the model has been trained yet """
        plt.figure(figsize=(20,4))
        plt.plot(np.arange(len(history)), history)
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
                plt.plot(np.delete(X, -1), Y)
                legends.append('layer %i (%s)' % (i, layer.__class__.__name__))

        plt.legend(legends)
        plt.title('activation distribution')

    def plot_gradients(self, bins: int=100) -> None:
        """ Plots neuron gradient distribution """
        plt.figure(figsize=(20,4))
        legends = []

        for i, layer in enumerate(self.layers):
            if layer.w is not None:
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
                plt.imshow(l.y[0, j, :, :], cmap='gray')
                plt.xlabel(f'kernel {str(j)}')

            plt.show()

    def __add_layer(self, layer, input_layer=False):
        if input_layer:
            self.layers.insert(0, layer)
        else:
            self.layers.append(layer)
            if layer.norm is not None:
                self.layers.append(layer.norm)
            if layer.activation is not None:
                self.layers.append(layer.activation)

        self.compiled = False

    def __forward(self):
        for layer in self.layers:
            layer.forward()

    def __backward(self):
        # set last layers gradient to be the loss gradient
        self.layers[-1].dy = self.loss_function.backward()
        layers_reversed = self.layers.copy()
        layers_reversed.reverse()

        for layer in layers_reversed:
            layer.backward()

    def __log(self, epoch, epochs, step, loss, verbose=False, val_loss=None):
        if verbose:
            if val_loss is not None:
                print('epoch %5s/%5s | time/epoch %.2f ms | loss %.4f | val_loss %.4f' % (epoch, epochs, step, loss, val_loss))
            else:
                print('epoch %5s/%5s | time/epoch %.2f ms | loss %.4f' % (epoch, epochs, step, loss))
        elif  epoch % (epochs // 10) == 0:
            if val_loss is not None:
                print('epoch %5s/%5s | time/epoch %.2f ms | loss %.4f | val_loss %.4f' % (epoch, epochs, step, loss, val_loss))
            else:
                print('epoch %5s/%5s | time/epoch %.2f ms | loss %.4f' % (epoch, epochs, step, loss))

    def __train(self):
        for l in self.layers:
            if l.mode is not None: l.mode = 'train'

    def __eval(self):
        for l in self.layers:
            if l.mode is not None: l.mode = 'eval'    

    def __check_dims(self, x, y=None):
        req_input_dim = self.layers[0].x.ndim

        if x.ndim != req_input_dim:
            raise Exception(f'Isput dimension must be {req_input_dim}.')
        
        if y is not None:
            req_output_dim = self.layers[-1].y.ndim

            if y.ndim != req_output_dim:
                raise Exception(f'Output dimension must be {req_output_dim}.')
