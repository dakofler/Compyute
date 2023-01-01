from model import utils, optimizers
import numpy as np
import time


class Network():
    def __init__(self) -> None:
        pass

    def predict (self, input):
        self.layers[0].i = input
    
    def summary(self):
        print(f'layer_type\tinput_shape\toutput_shape\tparameters')
        params = 0
        for l in self.layers:
            print(l.summary)
            if l.w is not None: params += np.size(l.w)
            if l.b is not None: params += np.size(l.b)
        print(f'total trainable parameters {params}')


class FeedForward(Network):
    def __init__(self, layers=[]) -> None:
        super().__init__()
        self.layers = []
        if len(layers) > 0:
            for layer in layers:
                self.add_layer(layer)

    def add_layer(self, layer):
        self.layers.append(layer)
    
    def compile(self, optimizer, loss, metric=None):
        self.optimizer = optimizer
        self.loss_function = loss
        self.metric = metric
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.integrate(i, None, self.layers[i + 1])
            elif i == len(self.layers) - 1:
                layer.integrate(i, self.layers[i - 1], None)
            else:
                layer.integrate(i, self.layers[i - 1], self.layers[i + 1])
    
    def __propagate(self):
        for layer in self.layers:
            layer.process()

    def train(self, x: np.ndarray, y: np.ndarray, epochs=100, batch_size = 0, log=True):
        if x.ndim != 4 or y.ndim != 2:
            print('Dimension must be 4 for input, 2 for output.')
            return
        batch_size = batch_size if batch_size > 0 else len(x)
        loss_hist = []

        for epoch in range(1, epochs + 1):
            epoch_loss_hist = []
            x_shuffled, y_shuffled = utils.shuffle(x, y)
            start = time.time()

            # train
            for i, p in enumerate(x_shuffled):
                if log: print(f'epoch {epoch}/{epochs}\tTraining ... {i + 1}/{batch_size}', end='\r')
                if i >= batch_size: break

                # compute loss
                loss, loss_gradient = self.loss_function(self.predict(p), np.squeeze(y_shuffled[i]))
                epoch_loss_hist.append(loss)

                # adjust weights
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

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        if self.metric is not None:
            loss, name, value, step = self.metric(x, y, self, self.loss_function)
            print (f'loss={round(loss, 4)}\t{name}={value}\ttime={step}ms')
        else:
            print('No metric defined.')

    def predict(self, input):
        if input.ndim != 3:
            print('Input shape must be of dim 3')
            return
        super().predict(input)
        self.__propagate()
        return self.layers[-1].o