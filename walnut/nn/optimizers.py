"""Parameter optimizers module"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from walnut import tensor_utils as tu
from walnut.tensor import Tensor


@dataclass()
class Optimizer(ABC):
    """Optimizer base class"""

    @abstractmethod
    def __call__(self, parameter: Tensor):
        ...


@dataclass
class SGD(Optimizer):
    """Implements the stochastic gradient descent algorithm.

    Parameters
    ----------
    l_r : float, optional
        Learning rate, by default 1e-2.
    momentum : float, optional
        Momentum factor, by default 0.
    nesterov : bool, optional
        Whether to use the neterov momentum algorithm, by default False.
    """

    l_r: float = 1e-2
    momentum: float = 0
    nesterov: bool = False

    def __call__(self, parameter: Tensor) -> None:
        """Updates a tensors data based on its gradients.

        Parameters
        ----------
        parameter : Tensor
            Tensor whose data is to be updated.
        """
        # get delta of previous updating cycle. If not availlable, initlaize with zeros.
        delta = parameter.params.get("delta", tu.zeros(parameter.data.shape).data)
        delta_new = -self.l_r * parameter.grad + self.momentum * delta

        if not self.nesterov:
            parameter.data = parameter.data + delta_new
        else:
            parameter.data = (
                parameter.data + self.momentum * delta_new - self.l_r * parameter.grad
            )

        parameter.params["delta"] = delta_new


@dataclass
class Adam(Optimizer):
    """Implements the adam algorithm according to Kingma et al., 2014.

    Parameters
    ----------
    l_r : float, optional
        Learning rate, by default 1e-3.
    beta1 : float, optional
        Exponential decay rate for the 1st momentum estimates, by default 0.9.
    beta2 : float, optional
        Exponential decay rate for the 2nd momentum estimates, by default 0.999.
    eps : float, optional
        Constant for numerical stability, by default 1e-07
    """

    l_r: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-07

    def __call__(self, parameter: Tensor):
        """Updates a tensors data based on its gradients.

        Parameters
        ----------
        parameter : Tensor
            Tensor whose data is to be updated.
        """
        # get momentum of previous updating cycle. If not availlable, initlaize with zeros.
        momentum = parameter.params.get("momentum", tu.zeros(parameter.data.shape).data)
        momentum_new = self.beta1 * momentum + (1 - self.beta1) * parameter.grad
        m_bc = momentum_new / (1 - self.beta1)
        parameter.params["momentum"] = momentum_new

        # get velocity of previous updating cycle. If not availlable, initlaize with zeros.
        velocity = parameter.params.get("velocity", tu.zeros(parameter.data.shape).data)
        velocity_new = self.beta2 * velocity + (1 - self.beta2) * parameter.grad**2
        v_bc = velocity_new / (1 - self.beta2)
        parameter.params["velocity"] = velocity_new
        parameter.data = parameter.data - m_bc * (self.l_r / (np.sqrt(v_bc) + self.eps))
