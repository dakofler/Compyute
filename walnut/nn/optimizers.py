"""Parameter optimizers module"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from walnut import tensor_utils as tu
from walnut.tensor import Tensor


@dataclass(slots=True)
class Optimizer(ABC):
    """Optimizer base class"""

    @abstractmethod
    def __call__(self, param: Tensor):
        ...


@dataclass(slots=True)
class SGD(Optimizer):
    """Updates parameters using stochastic gradient descent.

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

    def __call__(self, param: Tensor) -> None:
        """Updates parameters using stochastic gradient descent.

        Parameters
        ----------
        param : Tensor
            Tensor whose data is to be updated.
        """
        # get delta of previous updating cycle. If not availlable, initlaize with zeros.
        delta_prev = param.params.get("delta", tu.zeros(param.data.shape).data)
        delta = -self.l_r * param.grad + self.momentum * delta_prev

        if self.nesterov:
            delta = self.momentum * delta - self.l_r * param.grad

        param.data += delta
        param.params["delta"] = delta


@dataclass(slots=True)
class Adam(Optimizer):
    """Updates parameters following the adam learning algorithm as described by Kingma et al., 2014.

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

    def __call__(self, param: Tensor):
        """Updates parameters following the adam learning algorithm as described by Kingma et al., 2014.

        Parameters
        ----------
        parameter : Tensor
            Tensor whose data is to be updated.
        """
        # get momentum of previous updating cycle. If not availlable, initlaize with zeros.
        mom_prev = param.params.get("adam_mom", tu.zeros(param.data.shape).data)
        mom = self.beta1 * mom_prev + (1 - self.beta1) * param.grad
        m_bc = mom / (1 - self.beta1)
        param.params["adam_mom"] = mom

        # get velocity of previous updating cycle. If not availlable, initlaize with zeros.
        velo_prev = param.params.get("adam_velo", tu.zeros(param.data.shape).data)
        velo = self.beta2 * velo_prev + (1 - self.beta2) * param.grad**2
        v_bc = velo / (1 - self.beta2)
        param.params["adam_velo"] = velo

        delta = -m_bc * (self.l_r / (np.sqrt(v_bc) + self.eps))
        param.data += delta
        param.params["delta"] = delta
