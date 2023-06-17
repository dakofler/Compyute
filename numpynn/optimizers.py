"""parameter optimizers module"""

import numpy as np
from numpynn import layers


class Optimizer():
    """Optimizer base class"""

    def __init__(self, l_r):
        self.l_r = l_r


class SGD(Optimizer):
    """Implements the stochastic gradient descent algorithm.
    
    Args:
        l_r: Learning rate [optional].
        momentum: Momentum factor [optional].
        nesterov: If True, the neterov momentum algorithm is used [optional].
    """

    def __init__(self, l_r: float=1e-2, momentum: float=0.0, nesterov: bool=False) -> None:
        super().__init__(l_r)
        self.momentum = momentum
        self.nesterov = nesterov

    def __call__(self, model):
        for layer in model.layers:
            if not isinstance(layer, layers.ParamLayer):
                continue
            
            if layer.dw is not None:
                layer.w_delta = - self.l_r * layer.dw + self.momentum * layer.w_delta
                if not self.nesterov:
                    layer.w = layer.w + layer.w_delta
                else:
                    layer.w = layer.w + self.momentum * layer.w_delta - self.l_r * layer.dw

            if layer.db is not None:
                layer.b_delta = - self.l_r * layer.db + self.momentum * layer.b_delta
                if not self.nesterov:
                    layer.b = layer.b + layer.b_delta
                else:
                    layer.b = layer.b + self.momentum * layer.b_delta - self.l_r * layer.db


class Adam(Optimizer):
    """Implements the adam algorithm according to Kingma et al., 2014.
    
    Args:
        l_r: Learning rate [optional].
        beta1: Exponential decay rate for the 1st momentum estimates [optional].
        beta2: Exponential decay rate for the 2nd momentum estimates [optional].
        epsilon: Constant for numerical stability [optional].
    """

    def __init__(self, l_r: float=1e-3, beta1: float=0.9, beta2: float=0.999, epsilon: float=1e-07) -> None:
        super().__init__(l_r)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def __call__(self, model):
        for layer in model.layers:
            if not isinstance(layer, layers.ParamLayer):
                continue
            
            if layer.dw is not None:
                layer.w_m = self.beta1 * layer.w_m + (1 - self.beta1) * layer.dw
                m_bc = layer.w_m / (1 - self.beta1)
                layer.w_v = self.beta2 * layer.w_v + (1 - self.beta2) * np.power(layer.dw, 2)
                v_bc = layer.w_v / (1 - self.beta2)
                layer.w = layer.w - m_bc * (self.l_r / (np.sqrt(v_bc) + self.epsilon))

            if layer.db is not None:
                layer.b_m = self.beta1 * layer.b_m + (1 - self.beta1) * layer.db
                m_bc = layer.b_m / (1 - self.beta1)
                layer.b_v = self.beta2 * layer.b_v + (1 - self.beta2) * np.power(layer.db, 2)
                v_bc = layer.b_v / (1 - self.beta2)
                layer.b = layer.b - m_bc * (self.l_r / (np.sqrt(v_bc) + self.epsilon))
