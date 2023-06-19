"""parameter optimizers module"""

import numpy as np
from numpynn import layers, inits


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

            for param in layer.params:
                if param.delta is None:
                    self.__init_additional_param(param)

                param.delta = - self.l_r * param.grad + self.momentum * param.delta
                if not self.nesterov:
                    param.data = param.data + param.delta
                else:
                    param.data = param.data + self.momentum * param.delta - self.l_r * param.grad

    def __init_additional_param(self, param):
        param.delta = inits.zeros(param.data.shape)


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

            for param in layer.params:
                if param.mmtm is None:
                    self.__init_additional_param(param)

                param.mmtm = self.beta1 * param.mmtm + (1 - self.beta1) * param.grad
                m_bc = param.mmtm / (1 - self.beta1)
                param.velo = self.beta2 * param.velo + (1 - self.beta2) * param.grad**2
                v_bc = param.velo / (1 - self.beta2)
                param.data = param.data - m_bc * (self.l_r / (np.sqrt(v_bc) + self.epsilon))

    def __init_additional_param(self, param):
        param.mmtm = inits.zeros(param.data.shape)
        param.velo = inits.zeros(param.data.shape)
