"""Parameter optimizers module"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

from walnut import tensor_utils as tu
from walnut.tensor import Tensor


__all__ = ["SGD", "Adam"]


@dataclass(slots=True)
class Optimizer(ABC):
    """Optimizer base class"""

    parameters: list[Tensor] = field(default_factory=list)
    l_r: float = 1e-3

    @abstractmethod
    def step(self) -> None:
        """Updates parameters using their gradients."""


@dataclass(slots=True)
class SGD(Optimizer):
    """Updates parameters using stochastic gradient descent.

    Parameters
    ----------
    l_r : float, optional
        Learning rate, by default 1e-3.
    momentum : float, optional
        Momentum factor, by default 0.
    nesterov : bool, optional
        Whether to use the neterov momentum algorithm, by default False.
    """

    momentum: float = 0
    nesterov: bool = False

    def step(self) -> None:
        """Updates parameters using stochastic gradient descent."""
        for p in self.parameters:
            # get delta of previous updating cycle. If not availlable, initlaize with zeros.
            delta_prev = p.params.get("delta", tu.zeros(p.data.shape).data)
            delta = -self.l_r * p.grad + self.momentum * delta_prev

            if self.nesterov:
                delta = self.momentum * delta - self.l_r * p.grad

            p.data += delta
            p.params["delta"] = delta


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

    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-07

    def step(self):
        """Updates parameters following the adam learning algorithm as described by Kingma et al., 2014."""
        for p in self.parameters:
            # get momentum of previous updating cycle. If not availlable, initlaize with zeros.
            mom_prev = p.params.get("adam_mom", tu.zeros(p.data.shape).data)
            mom = self.beta1 * mom_prev + (1 - self.beta1) * p.grad
            m_bc = mom / (1 - self.beta1)
            p.params["adam_mom"] = mom

            # get velocity of previous updating cycle. If not availlable, initlaize with zeros.
            velo_prev = p.params.get("adam_velo", tu.zeros(p.data.shape).data)
            velo = self.beta2 * velo_prev + (1 - self.beta2) * p.grad**2
            v_bc = velo / (1 - self.beta2)
            p.params["adam_velo"] = velo

            delta = -m_bc * (self.l_r / (np.sqrt(v_bc) + self.eps))
            p.data += delta
            p.params["delta"] = delta
