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

            if layer.w is not None:
                layer.w.delta = - self.l_r * layer.w.grad + self.momentum * layer.w.delta
                if not self.nesterov:
                    layer.w.data = layer.w.data + layer.w.delta
                else:
                    layer.w.data = layer.w.data + self.momentum * layer.w.delta - self.l_r * layer.w.grad

            if layer.b is not None:
                layer.b.delta = - self.l_r * layer.b.grad + self.momentum * layer.b.delta
                if not self.nesterov:
                    layer.b.data = layer.b.data + layer.b.delta
                else:
                    layer.b.data = layer.b.data + self.momentum * layer.b.delta - self.l_r * layer.b.grad

            if layer.g is not None:
                layer.g.delta = - self.l_r * layer.g.grad + self.momentum * layer.g.delta
                if not self.nesterov:
                    layer.g.data = layer.g.data + layer.g.delta
                else:
                    layer.g.data = layer.g.data + self.momentum * layer.g.delta - self.l_r * layer.g.grad


class Adam(Optimizer):
    """Implements the adam algorithm according to Kingma et al., 2014.
    
    Args:
        l_r: Learning rate [optional].
        beta1: Exponential decay rate for the 1st momentum estimates [optional].
        beta2: Exponential decay rate for the 2nd momentum estimates [optional].
        epsilon: Constant for numerical stability [optional].
    """

    def __init__(self, l_r: float=1e-3, beta1: float=0.9, beta2: float=0.999,
                 epsilon: float=1e-07) -> None:
        super().__init__(l_r)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def __call__(self, model):
        for layer in model.layers:
            if not isinstance(layer, layers.ParamLayer):
                continue

            if layer.w is not None:
                layer.w.mmtm = self.beta1 * layer.w.mmtm + (1 - self.beta1) * layer.w.grad
                m_bc = layer.w.mmtm / (1 - self.beta1)
                layer.w.velo = self.beta2 * layer.w.velo + (1 - self.beta2) * layer.w.grad**2
                v_bc = layer.w.velo / (1 - self.beta2)
                layer.w = layer.w - m_bc * (self.l_r / (np.sqrt(v_bc) + self.epsilon))

            if layer.b is not None:
                layer.b.mmtm = self.beta1 * layer.b.mmtm + (1 - self.beta1) * layer.b.grad
                m_bc = layer.v.mmtm / (1 - self.beta1)
                layer.b.velo = self.beta2 * layer.b.velo + (1 - self.beta2) * layer.b.grad**2
                v_bc = layer.b.velo / (1 - self.beta2)
                layer.b = layer.b - m_bc * (self.l_r / (np.sqrt(v_bc) + self.epsilon))

            if layer.g is not None:
                layer.g.mmtm = self.beta1 * layer.g.mmtm + (1 - self.beta1) * layer.g.grad
                m_bc = layer.g.mmtm / (1 - self.beta1)
                layer.g.velo = self.beta2 * layer.g.velo + (1 - self.beta2) * layer.g.grad**2
                v_bc = layer.g.velo / (1 - self.beta2)
                layer.g = layer.g - m_bc * (self.l_r / (np.sqrt(v_bc) + self.epsilon))
