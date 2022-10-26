import numpy as np
import time
from model import optimizers

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
    
    def compile(self, optimizer=optimizers.error_dynamic):
        self.optimizer = optimizer
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

    def train(self, x: np.ndarray, y: np.ndarray, epochs=100, batch_size=0, learning_rate=0.5, momentum=0.0):
        if x.ndim != 3: return
        loss_hist = []
        batch_size = batch_size if batch_size > 0 else len(x)

        for epoch in range(1, epochs + 1):
            start = time.time()

            # compute learning rate
            learning_rate = self.optimizer(learning_rate, loss_hist, 10)

            shuffler = np.random.permutation(len(x))
            x_shuffled = x[shuffler]
            y_shuffled = y[shuffler]
            epoch_loss_hist = []

            # train
            for i, p in enumerate(x_shuffled):
                if i >= batch_size: break
                teaching_input = y_shuffled[i]

                # compute loss
                output = self.predict(p)
                sample_loss = teaching_input - output
                specific_sample_loss = 0.5 * np.linalg.norm(sample_loss) ** 2
                epoch_loss_hist.append(specific_sample_loss)

                # learn
                layers_reversed = self.layers.copy()
                layers_reversed.reverse()
                for j, l in enumerate(layers_reversed):
                    if j == 0:
                        l.learn(sample_loss)
                    else:
                        l.learn(learning_rate, momentum)
                 
            epoch_loss = sum(epoch_loss_hist) / len(epoch_loss_hist)
            loss_hist.append(epoch_loss)
            end = time.time()

            # validate
            # shuffler = np.random.permutation(batch_size)
            # x_shuffled = x[shuffler]
            # y_shuffled = y[shuffler]
            accuracy = self.__validate(x_shuffled, y_shuffled)

            end = time.time()
            print (f'epoch {epoch}/{epochs} loss={round(epoch_loss, 4)}\taccuracy={accuracy}\ttime/epoch={round((end - start) * 1000, 2)}ms\teta={learning_rate}')

    def __validate(self, x: np.ndarray, y: np.ndarray):
        c = 0

        for i, p in enumerate(x):
            val_input = y[i]
            p = self.predict(p)
            p[p == p.max()] = 1
            p[p != 1] = 0

            if not np.array_equal(val_input, p):
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