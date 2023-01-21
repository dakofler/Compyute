import numpy as np
from model.layers import Layer

class optimizer():
    def __init__(self) -> None:
        self.layers = None
        self.learning_rate = None
        self.momentum = None
        self.nesterov = None
        self.beta1 = None
        self.beta2 = None
        self.epsilon = None

    def optimize(self, loss_gradient: np.ndarray, model_layers: list[Layer]) -> None:
        layers_reversed = model_layers.copy()
        layers_reversed.reverse()
        layers_reversed[0].dy = loss_gradient
        self.layers = layers_reversed


class sgd(optimizer):
    def __init__(self, learning_rate: float=0.01, momentum: float=0.0, nesterov: bool=False) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

    def optimize(self, loss_gradient: np.ndarray, model_layers: list[Layer]) -> None:
        "Uses a layers gradients to adjust their weights and biases according to the stochastic gradient descent algorithm."
        super().optimize(loss_gradient, model_layers)

        for layer in self.layers:
            layer.learn()
            
            if layer.dw is not None:
                layer.w_change = - self.learning_rate * layer.dw + self.momentum * layer.w_change
                if not self.nesterov: layer.w = layer.w + layer.w_change
                else: layer.w = layer.w + self.momentum * layer.w_change - self.learning_rate * layer.dw
            
            if layer.db is not None:
                layer.b_change = - self.learning_rate * layer.db + self.momentum * layer.b_change
                if not self.nesterov: layer.b = layer.b + layer.b_change
                else: layer.b = layer.b + self.momentum * layer.b_change - self.learning_rate * layer.db


class adam(optimizer):
    def __init__(self, learning_rate: float=0.001, beta1: float=0.9, beta2: float=0.999, epsilon: float=1e-07) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def optimize(self, loss_gradient: np.ndarray, model_layers: list[Layer]) -> None:
        "Uses a layers gradients to adjust their weights and biases according to the adam algorithm."
        super().optimize(loss_gradient, model_layers)

        for layer in self.layers:
            layer.learn()

            if layer.dw is not None:
                layer.w_m = self.beta1 * layer.w_m + (1 - self.beta1) * layer.dw
                m_bc = layer.w_m / (1 - self.beta1)
                layer.w_v = self.beta2 * layer.w_v + (1 - self.beta2) * np.power(layer.dw, 2)
                v_bc = layer.w_v / (1 - self.beta2)
                layer.w = layer.w - m_bc * (self.learning_rate / (np.sqrt(v_bc) + self.epsilon))

            if layer.db is not None:
                layer.b_m = self.beta1 * layer.b_m + (1 - self.beta1) * layer.db
                m_bc = layer.b_m / (1 - self.beta1)
                layer.b_v = self.beta2 * layer.b_v + (1 - self.beta2) * np.power(layer.db, 2)
                v_bc = layer.b_v / (1 - self.beta2)
                layer.b = layer.b - m_bc * (self.learning_rate / (np.sqrt(v_bc) + self.epsilon))