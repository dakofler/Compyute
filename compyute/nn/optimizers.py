"""Parameter optimizers."""

from abc import ABC, abstractmethod
from typing import Iterator, Literal, Optional

from ..tensor_functions.transforming import tensorprod
from .parameter import Parameter

__all__ = ["SGD", "Adam", "AdamW", "NAdam"]


class Optimizer(ABC):
    """Optimizer base class.

    Parameters
    ----------
    parameters : Iterator[Parameter], optional
        Paramters to optimize. Defaults to ``None``.
    lr : float, optional
        Learning rate. Defaults to ``1e-3``.
    """

    def __init__(self, parameters: Optional[Iterator[Parameter]] = None, lr: float = 1e-3) -> None:
        self.lr = lr
        self.state: dict = {}
        self.t: int = 1

        if parameters is not None:
            self.parameters = parameters

    @property
    def parameters(self) -> Iterator[Parameter]:
        """Optimizer parameters."""
        return (p for p in self.state if isinstance(p, Parameter))

    @parameters.setter
    def parameters(self, value: Iterator[Parameter]) -> None:
        for p in value:
            self.state[p] = {}

    def reset_grads(self) -> None:
        """Resets parameter gradients to ``None``."""
        for p in self.parameters:
            p.grad = None

    @abstractmethod
    def step(self) -> None:
        """Updates parameters using their gradients."""


class SGD(Optimizer):
    r"""Updates parameters using Stochastic Gradient Descent (with optional momentum).

    .. math::
        \begin{aligned}
            & \rule{115mm}{1pt} \\
            & \textbf{Stochastic Gradient Descent} \\
            & \text{initialize: } v_0 \leftarrow 0 \\
            & \rule{115mm}{1pt} \\
            & \text{ 1: } \textbf{if } \lambda > 0 \\
            & \text{ 2: } \hspace{5mm} g_t \leftarrow g_t + \lambda p_{t-1} \\
            & \text{ 3: } \textbf{if } \mu = 0 \\
            & \text{ 4: } \hspace{5mm} p_t \leftarrow p_{t-1} - \eta g_t \\
            & \text{ 5: } \textbf{else} \\
            & \text{ 6: } \hspace{5mm} v_t \leftarrow \mu v_{t-1} - \eta g_t \\
            & \text{ 7: } \hspace{5mm} \textbf{if } nesterov \\
            & \text{ 8: } \hspace{10mm} p_t \leftarrow p_t + \mu v_t - \eta g_t \\
            & \text{ 9: } \hspace{5mm} \textbf{else} \\
            & \text{10: } \hspace{10mm} p_t \leftarrow p_{t-1} + v_t \\
            & \rule{115mm}{1pt} \\
        \end{aligned}

    where
        - :math:`p` ... parameter value
        - :math:`g` ... parameter gradient value
        - :math:`\eta` ... learning rate
        - :math:`\lambda` ... weight decay
        - :math:`\mu` ... momentum
        - :math:`v` ... velocity


    Parameters
    ----------
    parameters : Iterator[Parameter], optional
        Paramters to optimize. Defaults to ``None``.
    lr : float, optional
        Learning rate. Defaults to ``1e-3``.
    momentum : float, optional
        Momentum factor. Defaults to ``0``.
    nesterov : bool, optional
        Whether to use the neterov momentum algorithm. Defaults to ``False``.
    weight_decay : float, optional
        Weight decay factor. Defaults to ``0``.
    """

    def __init__(
        self,
        parameters: Optional[Iterator[Parameter]] = None,
        lr: float = 1e-3,
        momentum: float = 0,
        nesterov: bool = False,
        weight_decay: float = 0,
    ) -> None:
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay

    def step(self) -> None:
        """Updates parameters using stochastic gradient descent."""
        for p in self.parameters:
            if p.grad is None:
                continue

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
    r"""Updates parameters following the Adam learning algorithm
    as described by Kingma et al., 2014.

    .. math::
        \begin{aligned}
            & \rule{115mm}{1pt} \\
            & \textbf{Adam} \\
            & \text{initialize: } m_0 \leftarrow 0,  v_0 \leftarrow 0 \\
            & \rule{115mm}{1pt} \\
            & \text{ 1: } \textbf{if } \lambda > 0 \\
            & \text{ 2: } \hspace{5mm} g_t \leftarrow g_t + \lambda p_{t-1} \\
            & \text{ 3: } m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
            & \text{ 4: } v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
            & \text{ 5: } \hat{m_t} \leftarrow \frac{m_t}{1 - \beta_1^t} \\
            & \text{ 6: } \hat{v_t} \leftarrow \frac{v_t}{1 - \beta_2^t} \\
            & \text{ 7: } p_t \leftarrow p_{t-1} - \frac{\eta \hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon} \\
            & \rule{115mm}{1pt} \\
        \end{aligned}

    where
        - :math:`p` ... parameter value
        - :math:`g` ... parameter gradient value
        - :math:`\eta` ... learning rate
        - :math:`\lambda` ... weight decay
        - :math:`\beta_1` ... exponential decay rate for the 1st momentum estimate
        - :math:`\beta_2` ... exponential decay rate for the 2nd momentum estimate
        - :math:`\epsilon` ... constant for numerical stability
        - :math:`m` ... first moment estimate (exponential moving average)
        - :math:`v` ... second moment estimate (squared gradient)

    Parameters
    ----------
    parameters : Iterator[Parameter], optional
        Paramters to optimize. Defaults to ``None``.
    lr : float, optional
        Learning rate. Defaults to ``1e-3``.
    beta1 : float, optional
        Exponential decay rate for the 1st momentum estimate. Defaults to ``0.9``.
    beta2 : float, optional
        Exponential decay rate for the 2nd momentum estimate. Defaults to ``0.999``.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-08``.
    weight_decay : float, optional
        Weight decay factor. Defaults to ``0``.
    """

    def __init__(
        self,
        parameters: Optional[Iterator[Parameter]] = None,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self) -> None:
        for p in self.parameters:
            if p.grad is None:
                continue

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
    r"""Updates parameters following the AdamW learning algorithm.

    .. math::
        \begin{aligned}
            & \rule{115mm}{1pt} \\
            & \textbf{AdamW} \\
            & \text{initialize: } m_0 \leftarrow 0,  v_0 \leftarrow 0 \\
            & \rule{115mm}{1pt} \\
            & \text{ 1: } \textbf{if } \lambda > 0 \\
            & \text{ 2: } \hspace{5mm} p_t \leftarrow p_{t-1} - \eta \lambda p_{t-1} \\
            & \text{ 3: } m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
            & \text{ 4: } v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
            & \text{ 5: } \hat{m_t} \leftarrow \frac{m_t}{1 - \beta_1^t} \\
            & \text{ 6: } \hat{v_t} \leftarrow \frac{v_t}{1 - \beta_2^t} \\
            & \text{ 7: } p_t \leftarrow p_{t-1} - \frac{\eta \hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon} \\
            & \rule{115mm}{1pt} \\
        \end{aligned}

    where
        - :math:`p` ... parameter value
        - :math:`g` ... parameter gradient value
        - :math:`\eta` ... learning rate
        - :math:`\lambda` ... weight decay
        - :math:`\beta_1` ... exponential decay rate for the 1st momentum estimate
        - :math:`\beta_2` ... exponential decay rate for the 2nd momentum estimate
        - :math:`\epsilon` ... constant for numerical stability
        - :math:`m` ... first moment estimate (exponential moving average)
        - :math:`v` ... second moment estimate (squared gradient)

    Parameters
    ----------
    parameters : Iterator[Parameter], optional
        Paramters to optimize. Defaults to ``None``.
    lr : float, optional
        Learning rate. Defaults to ``1e-3``.
    beta1 : float, optional
        Exponential decay rate for the 1st momentum estimate. Defaults to ``0.9``.
    beta2 : float, optional
        Exponential decay rate for the 2nd momentum estimate. Defaults to ``0.999``.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-08``.
    weight_decay : float, optional
        Weight decay factor. Defaults to ``1e-2``.
    """

    def __init__(
        self,
        parameters: Optional[Iterator[Parameter]] = None,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
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
                p -= self.lr * self.weight_decay * p

            assert p.grad is not None  # just so pylance is happy

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
    r"""Updates parameters following the NAdam learning algorithm.

    .. math::
        \begin{aligned}
            & \rule{115mm}{1pt} \\
            & \textbf{NAdam} \\
            & \text{initialize: } m_0 \leftarrow 0,  v_0 \leftarrow 0 \\
            & \rule{115mm}{1pt} \\
            & \text{ 1: } \textbf{if } \lambda > 0 \\
            & \text{ 2: } \hspace{5mm} g_t \leftarrow g_t + \lambda p_{t-1} \\
            & \text{ 3: } \mu_t = \beta_1 \cdot (1 - \frac{1}{2} 0.96^{t \psi}) \\
            & \text{ 4: } \mu_{t+1} = \beta_1 \cdot (1 - \frac{1}{2} 0.96^{(t + 1) \psi}) \\
            & \text{ 5: } m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
            & \text{ 6: } v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
            & \text{ 7: } \hat{m_t} \leftarrow \frac{\mu_{t+1} m_t}{1 - \prod_{i = 1}^{t + 1} \mu_i} + \frac{1 - \mu_t}{1 - \prod_{i = 1}^{t} \mu_i} \\
            & \text{ 8: } \hat{v_t} \leftarrow \frac{v_t}{1 - \beta_2^t} \\
            & \text{ 9: } p_t \leftarrow p_{t-1} - \frac{\eta \hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon} \\
            & \rule{115mm}{1pt} \\
        \end{aligned}

    where
        - :math:`p` ... parameter value
        - :math:`g` ... parameter gradient value
        - :math:`\eta` ... learning rate
        - :math:`\lambda` ... weight decay
        - :math:`\psi` ... momentum decay
        - :math:`\beta_1` ... exponential decay rate for the 1st momentum estimate
        - :math:`\beta_2` ... exponential decay rate for the 2nd momentum estimate
        - :math:`\epsilon` ... constant for numerical stability
        - :math:`\mu` ... momentum coefficient
        - :math:`m` ... first moment estimate (exponential moving average)
        - :math:`v` ... second moment estimate (squared gradient)
        

    Parameters
    ----------
    parameters : Iterator[Parameter], optional
        Paramters to optimize. Defaults to ``None``.
    lr : float, optional
        Learning rate. Defaults to ``2e-3``.
    beta1 : float, optional
        Exponential decay rate for the 1st momentum estimate. Defaults to ``0.9``.
    beta2 : float, optional
        Exponential decay rate for the 2nd momentum estimate. Defaults to ``0.999``.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-08``.
    weight_decay : float, optional
        Weight decay factor. Defaults to ``0``.
    momentum_decay : float, optional
        Momentum decay factor. Defaults to ``4e-3``.
    """

    def __init__(
        self,
        parameters: Optional[Iterator[Parameter]] = None,
        lr: float = 2e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum_decay: float = 4e-3,
    ) -> None:
        super().__init__(parameters, lr)
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
