import numpy as np
import time
from model import losses

class Network():
    def __init__(self) -> None:
        pass

    def predict (self, input):
        self.layers[0].input = input


class FeedForward(Network):
    def __init__(self, layers=[]) -> None:
        super().__init__()
        self.layers = []
        if len(layers) > 0:
            for layer in layers:
                self.add_layer(layer)

    def add_layer(self, layer):
        self.layers.append(layer)
    
    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss
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

    def train(self, x: np.ndarray, y: np.ndarray, epochs=100, batch_size=0, log=True):
        if x.ndim != 3: return
        batch_size = batch_size if batch_size > 0 else len(x)
        loss_hist = []

        for epoch in range(1, epochs + 1):
            epoch_loss_hist = []

            # shuffle data
            shuffle_index = np.arange(len(x))
            np.random.shuffle(shuffle_index)
            x_shuffled = x[shuffle_index]
            y_shuffled = y[shuffle_index]

            start = time.time()

            # train
            for i, p in enumerate(x_shuffled):
                if i >= batch_size: break

                # compute loss
                loss, loss_gradient = self.loss(self.predict(p), np.squeeze(y_shuffled[i]))
                epoch_loss_hist.append(loss)

                # adjust weights
                self.optimizer.optimize(loss_gradient, self.layers)
            
            end = time.time()
            step = round((end - start) * 1000, 2)
            epoch_loss = sum(epoch_loss_hist) / len(epoch_loss_hist)
            loss_hist.append(epoch_loss)

            # validate
            # val_shuffle_index = np.arange(len(x))
            # np.random.shuffle(val_shuffle_index)
            # x_val_shuffled = x[val_shuffle_index]
            # y_val_shuffled = y[val_shuffle_index]

            accuracy = 0 #self.__validate(x_val_shuffled, y_val_shuffled)

            if log:
                print (f'epoch {epoch}/{epochs}\tloss={round(epoch_loss, 4)}\taccuracy={accuracy}%\ttime/epoch={step}ms')

    def __validate(self, x: np.ndarray, y: np.ndarray):
        c = 0

        for i, p in enumerate(x):
            prediction = self.predict(p)
            prediction[prediction == prediction.max()] = 1
            prediction[prediction != 1] = 0
            val_input = y[i].reshape(prediction.shape)
            if not np.array_equal(val_input, prediction):
                c = c + 1
        
        return round(100.0 / len(x) * (len(x) - c), 4)

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        accuracy = self.__validate(x, y)
        print (f'accuracy={accuracy}%')

    def predict(self, input):
        if input.ndim != 2: return
        super().predict(input)
        self.__propagate()
        return self.layers[-1].output