# neural network optimizers module

import numpy as np


class Optimizer():

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


class SGD(Optimizer):

    def __init__(self, learning_rate: float=1e-2, momentum: float=0.0, nesterov: bool=False) -> None:
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov

    def __call__(self, model):
        for layer in model.layers:
            if layer.dw is not None:
                layer.w_change = - self.learning_rate * layer.dw + self.momentum * layer.w_change
                if not self.nesterov: layer.w = layer.w + layer.w_change
                else: layer.w = layer.w + self.momentum * layer.w_change - self.learning_rate * layer.dw
            
            if layer.db is not None:
                layer.b_change = - self.learning_rate * layer.db + self.momentum * layer.b_change
                if not self.nesterov: layer.b = layer.b + layer.b_change
                else: layer.b = layer.b + self.momentum * layer.b_change - self.learning_rate * layer.db


class Adam(Optimizer):

    def __init__(self, learning_rate: float=1e-3, beta1: float=0.9, beta2: float=0.999, epsilon: float=1e-07) -> None:
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def __call__(self, model):
        for layer in model.layers:
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
