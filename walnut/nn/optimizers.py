"""Parameter optimizers module"""

from abc import ABC, abstractmethod
import numpy as np

from walnut.tensor import Tensor


__all__ = ["SGD", "Adam"]


class Optimizer(ABC):
    """Optimizer base class"""

    def __init__(self) -> None:
        self.parameters: list[Tensor] = []

    @abstractmethod
    def step(self) -> None:
        """Updates parameters using their gradients."""


class SGD(Optimizer):
    """Updates parameters using stochastic gradient descent."""

    def __init__(
        self,
        l_r: float = 1e-2,
        m: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        """Updates parameters using stochastic gradient descent.

        Parameters
        ----------
        l_r : float, optional
            Learning rate, by default 1e-2.
        m : float, optional
            Momentum factor, by default 0.
        nesterov : bool, optional
            Whether to use the neterov momentum algorithm, by default False.
        weight_deyas : float, optional
            Weight decay factor, by default 0.0.
        """
        super().__init__()
        self.l_r = l_r
        self.m = m
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.t: int = 1

    def step(self) -> None:
        """Updates parameters using stochastic gradient descent."""
        for p in self.parameters:
            if p.grad is None:
                continue

            if self.weight_decay > 0.0:
                p.grad += self.weight_decay * p.data

            if self.m > 0.0:
                if self.t > 1:
                    b_prev = p.params.get("sgd_b", np.zeros(p.data.shape))
                    b = self.m * b_prev + p.grad
                else:
                    b = p.grad

                p.params["sgd_b"] = b

                if self.nesterov:
                    p.grad += self.m * b
                else:
                    p.grad = b

            delta = -self.l_r * p.grad
            p.params["delta"] = delta  # for analysis
            p.data += delta

        self.t += 1


class Adam(Optimizer):
    """Updates parameters following the adam learning algorithm
    as described by Kingma et al., 2014."""

    def __init__(
        self,
        l_r: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """Updates parameters following the adam learning algorithm
        as described by Kingma et al., 2014.

        Parameters
        ----------
        l_r : float, optional
            Learning rate, by default 1e-3.
        beta1 : float, optional
            Exponential decay rate for the 1st momentum estimates, by default 0.9.
        beta2 : float, optional
            Exponential decay rate for the 2nd momentum estimates, by default 0.999.
        eps : float, optional
            Constant for numerical stability, by default 1e-07.
        weight_deyas : float, optional
            Weight decay factor, by default 0.0.
        """
        super().__init__()
        self.l_r = l_r
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t: int = 1

    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue

            if self.weight_decay > 0.0:
                p.grad += self.weight_decay * p.data

            m_prev = p.params.get("adam_m", np.zeros(p.data.shape))
            v_prev = p.params.get("adam_v", np.zeros(p.data.shape))

            m = self.beta1 * m_prev + (1.0 - self.beta1) * p.grad
            p.params["adam_m"] = m
            v = self.beta2 * v_prev + (1.0 - self.beta2) * p.grad**2
            p.params["adam_v"] = v

            m_hat = m / (1.0 - self.beta1**self.t)
            v_hat = v / (1.0 - self.beta2**self.t)

            delta = -self.l_r * m_hat / (v_hat**0.5 + self.eps)
            p.params["delta"] = delta  # for analysis
            p.data += delta

        self.t += 1
