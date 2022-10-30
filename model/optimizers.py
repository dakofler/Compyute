import numpy as np
from model import eta, activations, layers


class optimizer():
    def __init__(self) -> None:
        pass


class stochastic_gradient_descent(optimizer):
    def __init__(self, eta) -> None:
        super().__init__()
        self.eta = eta

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

                w = l.weights.copy()
                w = np.delete(w, -1, axis=1) # remove weights corresponding to bias neurons

                l.learn_output = np.dot(w.transpose(), delta) 
                l.delta_weights = - self.eta * np.append(l.prev_layer.output, [1.0], axis=0) * np.expand_dims(delta, 1)
                l.weights = l.weights + l.delta_weights