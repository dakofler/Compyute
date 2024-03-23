"""Parameter optimizers module"""

from abc import ABC, abstractmethod
from ..parameter import Parameter
from ...tensor import Tensor


__all__ = ["SGD", "Adam", "AdamW"]


class Optimizer(ABC):
    """Optimizer base class"""

    def __init__(self, parameters: list[Parameter] | None, lr: float) -> None:
        self.lr = lr
        self.state: dict[Parameter, dict[str, Tensor]] = {}
        self.t: int = 1

        if parameters is not None:
            self.parameters = parameters

    @property
    def parameters(self) -> list[Parameter]:
        """Optimizer parameters"""
        return list(self.state)

    @parameters.setter
    def parameters(self, value: list[Parameter]) -> None:
        for p in value:
            self.state[p] = {}

    @abstractmethod
    def step(self) -> None:
        """Updates parameters using their gradients."""


class SGD(Optimizer):
    """Updates parameters using stochastic gradient descent."""

    def __init__(
        self,
        parameters: list[Parameter] | None = None,
        lr: float = 1e-2,
        m: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        """Updates parameters using stochastic gradient descent.

        Parameters
        ----------
        parameters : list[Parameter] | None
            Paramters to optimize, by default None.
        lr : float, optional
            Learning rate, by default 1e-2.
        m : float, optional
            Momentum factor, by default 0.
        nesterov : bool, optional
            Whether to use the neterov momentum algorithm, by default False.
        weight_deyas : float, optional
            Weight decay factor, by default 0.0.
        """
        super().__init__(parameters, lr)
        self.m = m
        self.nesterov = nesterov
        self.weight_decay = weight_decay

    def step(self) -> None:
        """Updates parameters using stochastic gradient descent."""
        for p in self.parameters:
            if p.grad is None:
                continue

            g = p.grad.copy()

            if self.weight_decay > 0.0:
                g += self.weight_decay * p

            if self.m > 0.0:
                if self.t > 1:
                    b_prev = self.state[p].get("sgd_b", 0.0)
                    b = self.m * b_prev + g
                else:
                    b = g
                self.state[p]["sgd_b"] = b

                if self.nesterov:
                    g += self.m * b
                else:
                    g = b

            delta = -self.lr * g
            self.state[p]["delta"] = delta  # for analysis
            p.data += delta.data
        self.t += 1


class Adam(Optimizer):
    """Updates parameters following the Adam learning algorithm
    as described by Kingma et al., 2014."""

    def __init__(
        self,
        parameters: list[Parameter] | None = None,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """Updates parameters following the Adam learning algorithm
        as described by Kingma et al., 2014.

        Parameters
        ----------
        parameters : list[Parameter] | None
            Paramters to optimize, by default None.
        lr : float, optional
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
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self) -> None:
        for p in self.parameters:
            if p.grad is None:
                continue

            g = p.grad.copy()

            if self.weight_decay > 0.0:
                g += self.weight_decay * p

            m_prev = self.state[p].get("adam_m", 0.0)
            m = self.beta1 * m_prev + (1.0 - self.beta1) * g
            self.state[p]["adam_m"] = m

            v_prev = self.state[p].get("adam_v", 0.0)
            v = self.beta2 * v_prev + (1.0 - self.beta2) * g**2
            self.state[p]["adam_v"] = v

            m_hat = m / (1.0 - self.beta1**self.t)
            v_hat = v / (1.0 - self.beta2**self.t)

            delta = -self.lr * m_hat / (v_hat**0.5 + self.eps)
            self.state[p]["delta"] = delta  # for analysis
            p.data += delta.data
        self.t += 1


class AdamW(Optimizer):
    """Updates parameters following the AdamW learning algorithm."""

    def __init__(
        self,
        parameters: list[Parameter] | None = None,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """Updates parameters following the AdamW learning algorithm.

        Parameters
        ----------
        parameters : list[Parameter] | None
            Paramters to optimize, by default None.
        lr : float, optional
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
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self) -> None:
        for p in self.parameters:
            if p.grad is None:
                continue

            if self.weight_decay > 0:
                p.data -= self.lr * self.weight_decay * p.data

            m_prev = self.state[p].get("adam_m", 0.0)
            m = self.beta1 * m_prev + (1.0 - self.beta1) * p.grad
            self.state[p]["adam_m"] = m
            m_hat = m / (1.0 - self.beta1**self.t)

            v_prev = self.state[p].get("adam_v", 0.0)
            v = self.beta2 * v_prev + (1.0 - self.beta2) * p.grad**2
            self.state[p]["adam_v"] = v
            v_hat = v / (1.0 - self.beta2**self.t)

            delta = -self.lr * m_hat / (v_hat**0.5 + self.eps)
            self.state[p]["delta"] = delta  # for analysis
            p.data += delta.data
        self.t += 1
