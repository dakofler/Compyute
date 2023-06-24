"""parameter optimizers module"""

import numpy as np
from walnut.nn import layers
from walnut.tensor import zeros


class Optimizer():
    """Optimizer base class"""

    __slots__ = 'l_r'

    def __init__(self, l_r: float) -> None:
        self.l_r = l_r


class SGD(Optimizer):
    """Implements the stochastic gradient descent algorithm.
    
    ### Parameters
        l_r: `float`, optional
            Learning rate.
        momentum: `float`, optional
            Momentum factor.
        nesterov: `bool`, optional
            Whether to use the neterov momentum algorithm. By default it is not used.
    """

    def __init__(self, l_r: float = 1e-2, momentum: float = 0.0, nesterov: bool = False) -> None:
        super().__init__(l_r)
        self.momentum = momentum
        self.nesterov = nesterov

    def __call__(self, mdl_layers: list[layers.Layer]) -> None:
        """Updates the parametes of a given model.

        ### Parameters:
            mdl_layers: `list[Layer]`
                The model's layers whose parameters are updated.
        """
        for layer in mdl_layers:
            if not isinstance(layer, layers.ParamLayer):
                continue

            for param in layer.params:
                delta = param.params.get('delta', zeros(param.data.shape).data)
                delta_new = - self.l_r * param.grad + self.momentum * delta

                if not self.nesterov:
                    param.data = param.data + delta_new
                else:
                    param.data = param.data + self.momentum * delta_new - self.l_r * param.grad

                param.params['delta'] = delta_new


class Adam(Optimizer):
    """Implements the adam algorithm according to Kingma et al., 2014.
    
    ### Parameters
        l_r: `float`, optional
            Learning rate.
        beta1: `float`, optional
            Exponential decay rate for the 1st momentum estimates.
        beta2: `float`, optional
            Exponential decay rate for the 2nd momentum estimates.
        eps: `float`, optional
            Constant for numerical stability.
    """

    def __init__(self, l_r: float=1e-3, beta1: float=0.9, beta2: float=0.999,
                 eps: float=1e-07) -> None:
        super().__init__(l_r)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def __call__(self, mdl_layers: list[layers.Layer]):
        """Updates the parametes of a given model.

        ### Parameters:
            mdl_layers: `list[Layer]`
                The model's layers whose parameters are updated.
        """
        for layer in mdl_layers:
            if not isinstance(layer, layers.ParamLayer):
                continue

            for param in layer.params:
                momentum = param.params.get('momentum', zeros(param.data.shape).data)
                momentum_new = self.beta1 * momentum + (1 - self.beta1) * param.grad
                m_bc = momentum_new / (1 - self.beta1)
                param.params['momentum'] = momentum_new

                velocity = param.params.get('velocity', zeros(param.data.shape).data)
                velocity_new = self.beta2 * velocity + (1 - self.beta2) * param.grad**2
                v_bc = velocity_new / (1 - self.beta2)
                param.params['velocity'] = velocity_new

                param.data = param.data - m_bc * (self.l_r / (np.sqrt(v_bc) + self.eps))
