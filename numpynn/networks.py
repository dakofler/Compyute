"""neural network module"""

import time
import numpy as np
import matplotlib.pyplot as plt
from numpynn import layers, activations, utils, optimizers, losses, norms


class Sequential():
    """Feed forward neural network model"""

    def __init__(self, input_shape: tuple[int], layers: list[layers.Layer]) -> None:
        self.input_shape = input_shape
        self.layers = []

        for layer in layers:
            self.__add_layer(layer)

        self.optimizer = None
        self.loss_fn = None
        self.metric = None
        self.compiled = False

    def __call__(self, X: np.ndarray, Y: np.ndarray=None) -> None:
        """Computes an output and the loss based on imput samples.
        
        Args:
            x: Tensor of input values.
            y: Tensor of target values [optional].

        Returns:
            output: Tensor of output values.
            loss: Loss value, if target values are provided, else None.

        Raises:
            ShapeError: If input shape is not of dim 3.
        """
        self.__check_dims(X, Y)
        self.layers[0].x.data = X.astype('float32')
        self.__forward()
        output = self.layers[-1].y.data
        loss = None

        if Y is not None:
            loss = self.loss_fn(output, Y)
        return output, loss

    def compile(self, optimizer: optimizers.Optimizer, loss_fn: losses.Loss, metric) -> None:
        """ Compiles the model.
        
        Args:
            optimizer: Optimizer to be used to update weights and biases.
            loss_fn: Loss function to be used to compute the loss value and gradients.
            metric: Metric to be used to evaluate the model.
        """
        if not isinstance(self.layers[0], layers.Input):
            self.__add_layer(layers.Input(self.input_shape), input_layer=True)

        if not isinstance(self.layers[-1], layers.Output):
            self.__add_layer(layers.Output())

        for i, layer in enumerate(self.layers):
            if isinstance(layer, layers.Input):
                layer.compile(i, None, self.layers[i + 1])
            elif isinstance(layer, layers.Output):
                layer.compile(i, self.layers[i - 1], None)
            else:
                layer.compile(i, self.layers[i - 1], self.layers[i + 1])

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.compiled = True

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int=100, batch_size: int = None,
              verbose: bool=True, val_data: tuple[np.ndarray]=(None, None)) -> list[float]:
        """Trains the model using samples and targets.
        
        Args:
            X: Tensor of input feature values.
            Y: Tensor of target values.
            epochs: Number of training iterations [optional]
            batch_size: Number of samples used per epoch.
                If None, all training samples are used [optional].
            verbose: If false, printed output is decreased [optional].
            val_data: Data used for validation during training [optional].

        Returns:
            losses: List of loss values per epoch.

        Raises:
            ShapeError: If feature tensor is not of dim 4 or target tensor is not of dim 2. 
        """
        self.__check_dims(X, Y)
        history = []
        val_loss = None

        for epoch in range(1, epochs + 1):
            start = time.time()
            x_train, y_train = utils.shuffle(X, Y, batch_size)

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Applies the input to the model and returns it's predictions.
        
        Args:
            X: Tensor of input features.

        Returns:
            output: Tensor of predicted values.
        """
        self.__eval()
        pred, _ = self(X)
        return pred

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> None:
        """ Evaluates the model using a defined metric."""
        self.__eval()
        outputs, loss = self(X, Y)
        score = self.metric(outputs, Y)
        metric = self.metric.__name__
        print(f'loss {loss:.4f} | {metric:s}: {score:.4f}')

    def summary(self) -> None:
        """ Gives an overview of the model architecture.
        
        Raises:
            Error: If the model has not been compiled yet.
        """
        if not self.compiled:
            raise Exception('Model has not been compiled yet.')

        print(f'{"layer_type":15s} | {"input_shape":15s} | {"weight_shape":15s} | {"bias_shape":15s} | {"output_shape":15s} | {"parameters":15s}\n')
        sum_params = 0

        for layer in self.layers [1:-1]:
            if isinstance(layer, (layers.Input, layers.Output)):
                continue

            w_size = b_size = 0
            w_shape = b_shape = '-'

            if isinstance(layer, layers.ParamLayer) and layer.w.data is not None:
                w_size = np.size(layer.w.data)
                w_shape = str(layer.w.shape)

            if isinstance(layer, layers.ParamLayer) and layer.b.data is not None:
                b_size = np.size(layer.b.data)
                b_shape = str(layer.b.shape)

            params = w_size + b_size
            sum_params += params

            name = layer.__class__.__name__
            x_shape = str(layer.x.shape[1:])
            y_shape = str(layer.y.shape[1:])
            print(f'{name:15s} | {x_shape:15s} | {w_shape:15s} | {b_shape:15s} | {y_shape:15s} | {str(params):15s}')

        print(f'\ntotal trainable parameters {sum_params}')

    def plot_training_loss(self, history) -> None:
        """ Plots the loss over epochs."""
        plt.figure(figsize=(20,4))
        plt.plot(np.arange(len(history)), history)
        plt.xlabel('epoch')
        plt.ylabel('loss')

    def plot_activations(self, bins: int=100) -> None:
        """ Plots neuron activation distribution.
        
        Args:
            bins: Number of bins used in the histogram [optional].
        """
        plt.figure(figsize=(20,4))
        legends = []

        for i, layer in enumerate(self.layers):
            if isinstance(layer, activations.Activation) and not isinstance(layer, activations.Softmax):
                name = layer.__class__.__name__
                mean = layer.y.data.mean()
                std = layer.y.data.std()
                print(f'layer {i:d} ({name:s}) | mean {mean:.4f} | std {std:.4f}')
                y, x = np.histogram(layer.y.data, bins=bins)
                plt.plot(np.delete(x, -1), y)
                legends.append(f'layer {i:d} ({name:s})')

        plt.legend(legends)
        plt.title('activation distribution')

    def plot_gradients(self, bins: int=100) -> None:
        """ Plots neuron gradient distribution.
        
        Args:
            bins: Number of bins used in the histogram [optional].
        """
        plt.figure(figsize=(20,4))
        legends = []

        for i, layer in enumerate(self.layers):
            if isinstance(layer, layers.ParamLayer) and not isinstance(layer, norms.Layernorm):
                name = layer.__class__.__name__
                mean = layer.w.grad.mean()
                std = layer.w.grad.std()
                print(f'layer {i:d} ({name:s}) | mean {mean:.4f} | std {std:.4f}')
                y, x = np.histogram(layer.w.grad, bins=bins)
                x = np.delete(x, -1)
                plt.plot(x, y)
                legends.append(f'layer {i:d} ({name:s})')

        plt.legend(legends)
        plt.title('gradient distribution')

    def plot_conv_channels(self) -> None:
        """ Plots output channel activations convolutional layers."""
        conv_layers = [l for l in self.layers if isinstance(l, layers.Convolution)]

        if not conv_layers:
            print('No convolutional layers found.')

        for i,layer in enumerate(conv_layers):
            print(layer.__class__.__name__, i + 1)
            plt.figure(figsize=(40, 40))

            for j in range(layer.out_channels):
                plt.subplot(10, 8, j + 1)
                plt.imshow(layer.y.data[0, j, :, :], cmap='gray')
                plt.xlabel(f'out_channel {str(j)}')

            plt.show()

    def __add_layer(self, layer, input_layer=False):
        if input_layer:
            self.layers.insert(0, layer)
        else:
            self.layers.append(layer)

            if isinstance(layer, layers.ParamLayer):
                if layer.norm_fn is not None:
                    self.layers.append(layer.norm_fn)
                if layer.act_fn is not None:
                    self.layers.append(layer.act_fn)

        self.compiled = False

    def __forward(self):
        for layer in self.layers:
            layer.forward()

    def __backward(self):
        # set last layers gradient to be the loss gradient
        self.layers[-1].y.grad = self.loss_fn.backward()
        layers_reversed = self.layers.copy()
        layers_reversed.reverse()

        for layer in layers_reversed:
            layer.backward()

    def __log(self, epoch, epochs, step, loss, verbose=False, val_loss=None):
        def __log_line():
            line = f'epoch {epoch:5d}/{epochs:5d} | step {step:.2f} ms | loss {loss:.4f}'

            if val_loss is not None:
                line += f' | val_loss {val_loss:.4f}'

            print(line)

        if verbose:
            __log_line()
        elif epoch % (epochs // 10) == 0:
            __log_line()

    def __train(self):
        for layer in self.layers:
            if layer.mode is not None:
                layer.mode = 'train'

    def __eval(self):
        for layer in self.layers:
            if layer.mode is not None:
                layer.mode = 'eval'

    def __check_dims(self, X, Y=None):
        req_input_dim = self.layers[0].x.ndim

        if X.ndim != req_input_dim:
            raise Exception(f'Input dimension must be {req_input_dim}.')

        if Y is not None:
            req_output_dim = self.layers[-1].y.ndim

            if Y.ndim != req_output_dim:
                raise Exception(f'Output dimension must be {req_output_dim}.')
