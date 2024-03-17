"""Regularization layers module"""

from compyute.nn.module import Module
from compyute.random import multinomial
from compyute.tensor import Tensor
from compyute.types import ArrayLike


__all__ = ["Dropout"]


class Dropout(Module):
    """Dropout layer used to randomly reduce information and avoid overfitting."""

    def __init__(self, p: float = 0.5) -> None:
        """Dropout layer used to randomly reduce information and avoid overfitting.

        Parameters
        ----------
        p : float, optional
            Probability of values being set to zero, by default 0.5.
        """
        super().__init__()
        self.p = p

    def __repr__(self) -> str:
        name = self.__class__.__name__
        p = self.p
        return f"{name}({p=})"

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            p_comp = 1 - self.p
            choices = Tensor([0, 1], dtype=x.dtype, device=self.device)
            probs = Tensor([self.p, p_comp], dtype=x.dtype, device=self.device)
            d_map = multinomial(choices, probs, x.shape)
            y = x * d_map / p_comp

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)

                # use d_map as mask for grads
                return dy * d_map.data / p_comp

            self.backward = backward

        else:
            y = x

        self.set_y(y)
        return y
