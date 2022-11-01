import numpy as np
from model import learning_rate, activations, layers


class optimizer():
    def __init__(self) -> None:
        pass


class stochastic_gradient_descent(optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

    def optimize(self, loss_gradient, model_layers):
        layers_reversed = model_layers.copy()
        layers_reversed.reverse()

        for j, l in enumerate(layers_reversed):
            if j == 0:
                l.learn_output = loss_gradient
            elif j == len(layers_reversed) - 1:
                pass
            else:
                if type(l) != layers.Dense: continue

                prev_loss_gradient = l.succ_layer.learn_output # learn_output = error function gradient for output layer, sum (delta + w) for other layers

                # https://e2eml.school/softmax.html
                if l.activation == activations.Softmax:
                    d_softmax = l.activation(l.net, derivative=True)
                    prev_loss_gradient = np.reshape(prev_loss_gradient, (1, -1))
                    delta = np.squeeze(prev_loss_gradient @ d_softmax)
                else:
                    delta = l.activation(l.net, derivative=True) * prev_loss_gradient

                w = np.delete(l.weights.copy(), -1, axis=1) # remove weights corresponding to bias neurons
                l.learn_output = np.dot(w.transpose(), delta) # compute learning output for next layer, before weights are changed

                g = np.append(l.prev_layer.output, [1.0], axis=0) * np.expand_dims(delta, 1)
                l.delta_weights = - self.learning_rate * g + self.momentum * l.delta_weights

                # https://keras.io/api/optimizers/sgd/
                if not self.nesterov:
                    l.weights = l.weights + l.delta_weights
                else: 
                    l.weights = l.weights + self.momentum * l.delta_weights - self.learning_rate * g