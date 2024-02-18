"""Parameter optimizers module"""

from abc import ABC, abstractmethod

from walnut.nn.parameter import Parameter


__all__ = ["SGD", "Adam", "AdamW"]


class Optimizer(ABC):
    """Optimizer base class"""

    def __init__(self, lr: float) -> None:
        self.t: int = 1
        self.parameters: list[Parameter] = []
        self.lr = lr

    @abstractmethod
    def step(self) -> None:
        """Updates parameters using their gradients."""

    def reset_grads(self) -> None:
        """Resets parameter grads."""
        for parameter in self.parameters:
            parameter.grad = None

    def reset_optimizer_params(self) -> None:
        """Resets temporary values used in the step method."""
        for parameter in self.parameters:
            parameter.optimizer_params = {}


class SGD(Optimizer):
    """Updates parameters using stochastic gradient descent."""

    def __init__(
        self,
        lr: float = 1e-2,
        m: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        """Updates parameters using stochastic gradient descent.

        Parameters
        ----------
        lr : float, optional
            Learning rate, by default 1e-2.
        m : float, optional
            Momentum factor, by default 0.
        nesterov : bool, optional
            Whether to use the neterov momentum algorithm, by default False.
        weight_deyas : float, optional
            Weight decay factor, by default 0.0.
        """
        super().__init__(lr)
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
                g += self.weight_decay * p.data

            if self.m > 0.0:
                if self.t > 1:
                    b_prev = p.optimizer_params.get("sgd_b", 0.0)
                    b = self.m * b_prev + g
                else:
                    b = g
                p.optimizer_params["sgd_b"] = b

                if self.nesterov:
                    g += self.m * b
                else:
                    g = b

            delta = -self.lr * g
            p.optimizer_params["delta"] = delta  # for analysis
            p.data += delta
        self.t += 1


class Adam(Optimizer):
    """Updates parameters following the Adam learning algorithm
    as described by Kingma et al., 2014."""

    def __init__(
        self,
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
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue

            g = p.grad.copy()

            if self.weight_decay > 0.0:
                g += self.weight_decay * p.data

            m_prev = p.optimizer_params.get("adam_m", 0.0)
            m = self.beta1 * m_prev + (1.0 - self.beta1) * g
            p.optimizer_params["adam_m"] = m

            v_prev = p.optimizer_params.get("adam_v", 0.0)
            v = self.beta2 * v_prev + (1.0 - self.beta2) * g**2
            p.optimizer_params["adam_v"] = v

            m_hat = m / (1.0 - self.beta1**self.t)
            v_hat = v / (1.0 - self.beta2**self.t)

            delta = -self.lr * m_hat / (v_hat**0.5 + self.eps)
            p.optimizer_params["delta"] = delta  # for analysis
            p.data += delta
        self.t += 1


class AdamW(Optimizer):
    """Updates parameters following the AdamW learning algorithm."""

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """Updates parameters following the AdamW learning algorithm.

        Parameters
        ----------
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
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue

            if self.weight_decay > 0:
                p.data -= self.lr * self.weight_decay * p.data

            m_prev = p.optimizer_params.get("adam_m", 0)
            m = self.beta1 * m_prev + (1.0 - self.beta1) * p.grad
            p.optimizer_params["adam_m"] = m
            m_hat = m / (1.0 - self.beta1**self.t)

            v_prev = p.optimizer_params.get("adam_v", 0)
            v = self.beta2 * v_prev + (1.0 - self.beta2) * p.grad**2
            p.optimizer_params["adam_v"] = v
            v_hat = v / (1.0 - self.beta2**self.t)

            delta = -self.lr * m_hat / (v_hat**0.5 + self.eps)
            p.optimizer_params["delta"] = delta  # for analysis
            p.data += delta
        self.t += 1
