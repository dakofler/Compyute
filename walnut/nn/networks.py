"""neural network module"""

import time
import numpy as np
import matplotlib.pyplot as plt
from walnut.nn import optimizers
from walnut.nn import activations, layers, losses, norms
from walnut.tensor import Tensor, shuffle


class Sequential():
    """Feed forward neural network model.
    
    ### Parameters
        input_shape: `tuple[int]`
            Shape of input tensor ignoring axis 0 (batches).
        mdl_layers: `list[Layer]`
            Layers of the neural network model.
    """

    __slots__ = 'input_shape', 'mdl_layers', 'optimizer', 'loss_fn', 'metric', 'compiled'

    def __init__(self, input_shape: tuple[int], mdl_layers: list[layers.Layer]) -> None:
        self.input_shape = input_shape
        self.mdl_layers = []

        for layer in mdl_layers:
            self.__add_layer(layer)

        self.optimizer = None
        self.loss_fn = None
        self.metric = None
        self.compiled = False

    def __call__(self, X: Tensor, Y: Tensor = None) -> None:
        """Computes an output and the loss based on imput samples.
        
        ### Parameters
            x: `Tensor`
                Tensor of input values.
            y: `Tensor`, optional
                Tensor of target values. If provided, a loss is also computed and returned.

        ### Returns
            output: `Tensor`
                Tensor of predicted values.
            loss: 'Tensor' or None
                Loss value, if target values are provided.

        ### Raises
            ValueError:
                If input shape is not of dim 3.
        """
        self.__check_dims(X, Y)
        self.mdl_layers[0].x.data = X.data
        self.__forward()
        output = self.mdl_layers[-1].y
        loss = None

        if Y is not None:
            loss = self.loss_fn(output, Y)
        return output, loss

    def compile(self, optimizer: optimizers.Optimizer, loss_fn: losses.Loss, metric) -> None:
        """ Compiles the model.
        
        ### Parameters
            optimizer: `Optimizer`
                Optimizer algorithm to be used to update parameters.
            loss_fn: `Loss`
                Loss function to be used to compute losses and gradients.
            metric:
                Metric to be used to evaluate the model.
        """
        if not isinstance(self.mdl_layers[0], layers.Input):
            self.__add_layer(layers.Input(self.input_shape), input_layer=True)

        if not isinstance(self.mdl_layers[-1], layers.Output):
            self.__add_layer(layers.Output())

        for i, layer in enumerate(self.mdl_layers):
            if isinstance(layer, layers.Input):
                layer.compile(i, None, self.mdl_layers[i + 1])
            elif isinstance(layer, layers.Output):
                layer.compile(i, self.mdl_layers[i - 1], None)
            else:
                layer.compile(i, self.mdl_layers[i - 1], self.mdl_layers[i + 1])

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.compiled = True

    def train(self, X: Tensor, Y: Tensor, epochs: int=100, batch_size: int = None,
              verbose: bool=True, val_data: tuple[Tensor]=(None, None)) -> list[float]:
        """Trains the model using samples and targets.
        
        ### Parameters
            X: `Tensor`
                Tensor of input values (features).
            Y: `Tensor`
                Tensor of target values.
            epochs: `int`, optional
                Number of training iterations. By default, training lasts for 100 iterations.
            batch_size: `int`
                Number of training samples used per epoch.
                By default, all training samples are used.
            verbose: `bool`
                Whether to print out intermediate results while training.
                By default results are printed.
            val_data: `tuple[Tensor]`, optional
                Data used for validation during training.
                By default, validation is inactive during training.

        ### Returns
            losses: `list[float]`
                List of loss values per epoch.

        ### Raises
            ValueError:
                If input or target tensor dims do not match model dims. 
        """
        self.__check_dims(X, Y)
        history = []
        val_loss = None

        for epoch in range(1, epochs + 1):
            start = time.time()
            x_train, y_train = shuffle(X, Y, batch_size)

            # training
            self.__train()
            _, loss = self(x_train, y_train)
            self.__backward()
            self.optimizer(self.mdl_layers)

            # validation
            self.__eval()

            if val_data[0] is not None:
                _, val_loss = self(*val_data)

            end = time.time()
            step = round((end - start) * 1000.0, 2)

            if verbose:
                self.__log(epoch, epochs, step, loss, val_loss)

            history.append(loss.item())

        return history

    def predict(self, X: Tensor) -> Tensor:
        """Applies the input to the model and returns it's predictions.
        
        ### Parameters
            X: `Tensor`
                Tensor of input features.

        ### Returns
            predictions: `Tensor`
                Tensor of predicted values.
        """
        self.__eval()
        pred, _ = self(X)
        return pred

    def evaluate(self, X: Tensor, Y: Tensor) -> None:
        """ Evaluates the model using a defined metric.
        
        ### Parameters
            X: `Tensor`
                Tensor of input values (features).
            Y: `Tensor`
                Tensor of target values.
        """
        self.__eval()
        outputs, loss = self(X, Y)
        score = self.metric(outputs, Y)
        metric = self.metric.__name__
        print(f'loss {loss.item():.4f} | {metric:s}: {score:.4f}')

    def summary(self) -> None:
        """ Gives an overview of the model architecture.
        
        ### Raises
            Exception:
                If the model has not been compiled yet.
        """
        if not self.compiled:
            raise Exception('Model has not been compiled yet.')

        print(f'{"layer_type":15s} | {"input_shape":15s} | {"weight_shape":15s} | '
              + f'{"bias_shape":15s} | {"output_shape":15s} | {"parameters":15s}\n')
        sum_params = 0

        for layer in self.mdl_layers[1:-1]:
            if isinstance(layer, (layers.Input, layers.Output)):
                continue

            w_size = b_size = 0
            w_shape = b_shape = '-'

            if isinstance(layer, layers.ParamLayer) and layer.w is not None:
                w_size = np.size(layer.w.data)
                w_shape = str(layer.w.shape)

            if isinstance(layer, layers.ParamLayer) and layer.b is not None:
                b_size = np.size(layer.b.data)
                b_shape = str(layer.b.shape)

            params = w_size + b_size
            sum_params += params

            name = layer.__class__.__name__
            x_shape = str(layer.x.shape[1:])
            y_shape = str(layer.y.shape[1:])
            print(f'{name:15s} | {x_shape:15s} | {w_shape:15s} | '
                  + f'{b_shape:15s} | {y_shape:15s} | {str(params):15s}')

        print(f'\ntotal trainable parameters {sum_params}')

    def plot_training_loss(self, loss_hist: list[float]) -> None:
        """ Plots the loss over epochs.
        
        ### Parameters
            loss_hist: `list[float]`
                List of loss values per epoch.
        """
        plt.figure(figsize=(20,4))
        plt.plot(np.arange(len(loss_hist)), loss_hist)
        plt.xlabel('epoch')
        plt.ylabel('loss')

    def plot_activations(self, bins: int = 100) -> None:
        """ Plots neuron activation distribution.
        
        ### Parameters
            bins: `int`, optional
                Number of bins used in the histogram. By default, 100 bins are used.
        """
        plt.figure(figsize=(20,4))
        legends = []

        for i, layer in enumerate(self.mdl_layers):
            if (isinstance(layer, activations.Activation)
                    and not isinstance(layer, activations.Softmax)):
                name = layer.__class__.__name__
                mean = layer.y.data.mean()
                std = layer.y.data.std()
                print(f'layer {i:d} ({name:s}) | mean {mean:.4f} | std {std:.4f}')
                y_vals, x_vals = np.histogram(layer.y.data, bins=bins)
                plt.plot(np.delete(x_vals, -1), y_vals)
                legends.append(f'layer {i:d} ({name:s})')

        plt.legend(legends)
        plt.title('activation distribution')

    def plot_gradients(self, bins: int = 100) -> None:
        """ Plots neuron gradient distribution.
        
        ### Parameters
            bins: `int`, optional
                Number of bins used in the histogram. By default, 100 bins are used.
        """
        plt.figure(figsize=(20,4))
        legends = []

        for i, layer in enumerate(self.mdl_layers):
            if isinstance(layer, layers.ParamLayer) and not isinstance(layer, norms.Layernorm):
                name = layer.__class__.__name__
                mean = layer.w.grad.mean()
                std = layer.w.grad.std()
                print(f'layer {i:d} ({name:s}) | mean {mean:.4f} | std {std:.4f}')
                y_vals, x_vals = np.histogram(layer.w.grad, bins=bins)
                x_vals = np.delete(x_vals, -1)
                plt.plot(x_vals, y_vals)
                legends.append(f'layer {i:d} ({name:s})')

        plt.legend(legends)
        plt.title('gradient distribution')

    def plot_conv_channels(self) -> None:
        """ Plots output channel activations convolutional layers.
        
        ### Raises
            Exception:
                If the model does not contain conv layers.
        """
        conv_layers = [l for l in self.mdl_layers if isinstance(l, layers.Convolution)]

        if not conv_layers:
            raise Exception('no convolutional layers found')

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
            self.mdl_layers.insert(0, layer)
        else:
            self.mdl_layers.append(layer)

            if isinstance(layer, layers.ParamLayer):
                if layer.norm_fn is not None:
                    self.mdl_layers.append(layer.norm_fn)
                if layer.act_fn is not None:
                    self.mdl_layers.append(layer.act_fn)

        self.compiled = False

    def __forward(self):
        for layer in self.mdl_layers:
            layer.forward()

    def __backward(self):
        # set last layers gradient to be the loss gradient
        self.mdl_layers[-1].y.grad = self.loss_fn.backward().data
        layers_reversed = self.mdl_layers.copy()
        layers_reversed.reverse()

        for layer in layers_reversed:
            layer.backward()

    def __log(self, epoch, epochs, step, loss, val_loss=None):
        def __log_line():
            line = f'epoch {epoch:5d}/{epochs:5d} | step {step:.2f} ms | loss {loss.item():.4f}'

            if val_loss is not None:
                line += f' | val_loss {val_loss.item():.4f}'

            print(line)

        if epochs < 10:
            __log_line()
        elif epoch % (epochs // 10) == 0:
            __log_line()

    def __train(self):
        for layer in self.mdl_layers:
            if layer.mode is not None:
                layer.mode = 'train'

    def __eval(self):
        for layer in self.mdl_layers:
            if layer.mode is not None:
                layer.mode = 'eval'

    def __check_dims(self, x: Tensor, y: Tensor = None):
        req_input_dim = self.mdl_layers[0].x.ndim

        if x.ndim != req_input_dim:
            raise ValueError(f'Input dimension must be {req_input_dim}.')

        if y is not None:
            req_output_dim = self.mdl_layers[-1].y.ndim

            if y.ndim != req_output_dim:
                raise ValueError(f'Output dimension must be {req_output_dim}.')
