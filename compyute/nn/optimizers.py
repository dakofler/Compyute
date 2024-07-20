"""Parameter optimizers module"""

from abc import ABC, abstractmethod
from typing import Iterator, Literal, Optional

from ..tensor_functions.computing import tensorprod
from ..tensor_functions.transforming import clip
from .parameter import Parameter

__all__ = ["SGD", "Adam", "AdamW", "NAdam"]


class Optimizer(ABC):
    """Optimizer base class"""

    def __init__(
        self,
        parameters: Optional[Iterator[Parameter]],
        lr: float,
        grad_clip_value: Optional[float] = None,
    ) -> None:
        self.lr = lr
        self.state: dict = {}
        self.t: int = 1
        self.grad_clip_value = grad_clip_value

        if parameters is not None:
            self.parameters = parameters

    @property
    def parameters(self) -> Iterator[Parameter]:
        """Optimizer parameters"""
        return (p for p in self.state if isinstance(p, Parameter))

    @parameters.setter
    def parameters(self, value: Iterator[Parameter]) -> None:
        for p in value:
            self.state[p] = {}

    def reset_grads(self) -> None:
        """Resets parameter gradients."""
        for p in self.parameters:
            p.grad = None

    @abstractmethod
    def step(self) -> None:
        """Updates parameters using their gradients."""


class SGD(Optimizer):
    """Updates parameters using stochastic gradient descent."""

    def __init__(
        self,
        parameters: Optional[Iterator[Parameter]] = None,
        lr: float = 0.001,
        momentum: float = 0,
        nesterov: bool = False,
        weight_decay: float = 0,
        grad_clip_value: Optional[float] = None,
    ) -> None:
        """Updates parameters using stochastic gradient descent (with momentum).

        Parameters
        ----------
        parameters : Iterator[Parameter], optional
            Paramters to optimize, by default None.
        lr : float, optional
            Learning rate, by default 0.001.
        momentum : float, optional
            Momentum factor, by default 0.
        nesterov : bool, optional
            Whether to use the neterov momentum algorithm, by default False.
        weight_deyas : float, optional
            Weight decay factor, by default 0.
        grad_clip_value : float, optional
            Clips the gradient of all parameter to the set value, by default None.
        """
        super().__init__(parameters, lr, grad_clip_value)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay

    def step(self) -> None:
        """Updates parameters using stochastic gradient descent."""
        for p in self.parameters:
            if p.grad is None:
                continue

            # gradient clipping
            if self.grad_clip_value is not None:
                p.grad = clip(p.grad, -self.grad_clip_value, self.grad_clip_value)

            grad = p.grad.copy()

            if self.weight_decay > 0:
                grad += self.weight_decay * p

            if self.momentum == 0:
                p -= self.lr * grad
            else:
                prev_v = self.state[p].get("v", 0)
                v = self.momentum * prev_v - self.lr * grad  # velocity
                self.state[p]["v"] = v

                if self.nesterov:
                    p += self.momentum * v - self.lr * grad
                else:
                    p += v

        self.t += 1


class Adam(Optimizer):
    """Updates parameters following the Adam learning algorithm
    as described by Kingma et al., 2014."""

    def __init__(
        self,
        parameters: Optional[Iterator[Parameter]] = None,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0,
        grad_clip_value: Optional[float] = None,
    ) -> None:
        """Updates parameters following the Adam learning algorithm
        as described by Kingma et al., 2014.

        Parameters
        ----------
        parameters : Iterator[Parameter], optional
            Paramters to optimize, by default None.
        lr : float, optional
            Learning rate, by default 0.001.
        beta1 : float, optional
            Exponential decay rate for the 1st momentum estimates, by default 0.9.
        beta2 : float, optional
            Exponential decay rate for the 2nd momentum estimates, by default 0.999.
        eps : float, optional
            Constant for numerical stability, by default 1e-08.
        weight_deyas : float, optional
            Weight decay factor, by default 0.
        grad_clip_value : float, optional
            Clips the gradient of all parameter to the set value, by default None.
        """
        super().__init__(parameters, lr, grad_clip_value)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self) -> None:
        for p in self.parameters:
            if p.grad is None:
                continue

            # gradient clipping
            if self.grad_clip_value is not None:
                p.grad = clip(p.grad, -self.grad_clip_value, self.grad_clip_value)

            grad = p.grad.copy()

            if self.weight_decay > 0:
                grad += self.weight_decay * p

            # first moment estimate (exponential moving average)
            prev_m = self.state[p].get("m", 0)
            m = self.beta1 * prev_m + (1 - self.beta1) * grad
            self.state[p]["m"] = m

            # second moment estimate (squared gradient)
            prev_v = self.state[p].get("v", 0)
            v = self.beta2 * prev_v + (1 - self.beta2) * grad**2
            self.state[p]["v"] = v

            m /= 1 - self.beta1**self.t
            v /= 1 - self.beta2**self.t

            p -= self.lr * m / (v**0.5 + self.eps)

        self.t += 1


class AdamW(Optimizer):
    """Updates parameters following the AdamW learning algorithm."""

    def __init__(
        self,
        parameters: Optional[Iterator[Parameter]] = None,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        grad_clip_value: Optional[float] = None,
    ) -> None:
        """Updates parameters following the AdamW learning algorithm.

        Parameters
        ----------
        parameters : Iterator[Parameter], optional
            Paramters to optimize, by default None.
        lr : float, optional
            Learning rate, by default 0.001.
        beta1 : float, optional
            Exponential decay rate for the 1st momentum estimates, by default 0.9.
        beta2 : float, optional
            Exponential decay rate for the 2nd momentum estimates, by default 0.999.
        eps : float, optional
            Constant for numerical stability, by default 1e-08.
        weight_decay : float, optional
            Weight decay factor, by default 0.
        grad_clip_value : float, optional
            Clips the gradient of all parameter to the set value, by default None.
        """
        super().__init__(parameters, lr, grad_clip_value)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self) -> None:
        for p in self.parameters:
            if p.grad is None:
                continue

            # gradient clipping
            if self.grad_clip_value is not None:
                p.grad = clip(p.grad, -self.grad_clip_value, self.grad_clip_value)

            if self.weight_decay > 0:
                p.data -= self.lr * self.weight_decay * p.data

            # first moment estimate (exponential moving average)
            prev_m = self.state[p].get("m", 0)
            m = self.beta1 * prev_m + (1 - self.beta1) * p.grad
            self.state[p]["m"] = m

            # second moment estimate (squared gradient)
            prev_v = self.state[p].get("v", 0)
            v = self.beta2 * prev_v + (1 - self.beta2) * p.grad**2
            self.state[p]["v"] = v

            m /= 1 - self.beta1**self.t
            v /= 1 - self.beta2**self.t

            p -= self.lr * m / (v**0.5 + self.eps)

        self.t += 1


class NAdam(Optimizer):
    """Updates parameters following the NAdam learning algorithm."""

    def __init__(
        self,
        parameters: Optional[Iterator[Parameter]] = None,
        lr: float = 0.002,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum_decay: float = 0.004,
        grad_clip_value: Optional[float] = None,
    ) -> None:
        """Updates parameters following the NAdam learning algorithm.

        Parameters
        ----------
        parameters : Iterator[Parameter], optional
            Paramters to optimize, by default None.
        lr : float, optional
            Learning rate, by default 0.002.
        beta1 : float, optional
            Exponential decay rate for the 1st momentum estimates, by default 0.9.
        beta2 : float, optional
            Exponential decay rate for the 2nd momentum estimates, by default 0.999.
        eps : float, optional
            Constant for numerical stability, by default 1e-08.
        weight_decay : float, optional
            Weight decay factor, by default 0.
        momentum_decay : float, optional
            Momentum decay factor, by default 0.004.
        grad_clip_value : float, optional
            Clips the gradient of all parameter to the set value, by default None.
        """
        super().__init__(parameters, lr, grad_clip_value)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum_decay = momentum_decay
        self.state["mus"] = []

    def step(self) -> None:
        # momentum coefficient
        mu = self.beta1 * (1 - 0.5 * 0.96 ** (self.t * self.momentum_decay))
        self.state["mus"].append(mu)
        mu_prod = tensorprod(self.state["mus"])
        next_mu = self.beta1 * (1 - 0.5 * 0.96 ** ((self.t + 1) * self.momentum_decay))

        for p in self.parameters:
            if p.grad is None:
                continue

            # gradient clipping
            if self.grad_clip_value is not None:
                p.grad = clip(p.grad, -self.grad_clip_value, self.grad_clip_value)

            grad = p.grad.copy()

            if self.weight_decay > 0:
                grad += self.weight_decay * p

            # first moment estimate (exponential moving average)
            prev_m = self.state[p].get("m", 0)
            m = self.beta1 * prev_m + (1 - self.beta1) * grad
            self.state[p]["m"] = m

            # second moment estimate (squared gradient)
            prev_v = self.state[p].get("v", 0)
            v = self.beta2 * prev_v + (1 - self.beta2) * grad**2
            self.state[p]["v"] = v

            m_hat = next_mu * m / (1 - mu_prod * next_mu) + (1 - mu) * grad / (1 - mu_prod)
            v /= 1 - self.beta2**self.t

            p -= self.lr * m_hat / (v**0.5 + self.eps)

        self.t += 1


_OptimizerLike = Optimizer | Literal["sgd", "adam", "adamw", "nadam"]
OPTIMIZERS = {"sgd": SGD, "adam": Adam, "adamw": AdamW, "nadam": NAdam}


def get_optimizer(optimizer: _OptimizerLike) -> Optimizer:
    """Returns an instance of an optimizer."""
    if isinstance(optimizer, Optimizer):
        return optimizer
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {optimizer}.")
    return OPTIMIZERS[optimizer]()
