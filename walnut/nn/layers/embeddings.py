"""Embedding layers module"""

from dataclasses import dataclass
import numpy as np

from walnut.tensor import ShapeLike
from walnut.nn.optimizers import Optimizer
from walnut.nn import inits
from walnut.nn.inits import Init
from walnut.nn.layers.parameter import ParamLayer


@dataclass(init=False, repr=False)
class Character(ParamLayer):
    """Character level embedding layer."""

    def __init__(
        self,
        out_channels: int,
        init: str = "kaiming_he",
        input_shape: ShapeLike | None = None,
    ) -> None:
        """Single character embedding layer.

        Parameters
        ----------
        out_channels : int
            Number of output channels (embedding dimensions) of the layer.
        init : str, optional
            Initialization function for weights, by default "kaiming_he".
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the layer is used as input, by default None.
        """
        super().__init__(
            init_fn_name=init,
            input_shape=input_shape,
        )
        self.out_channels = out_channels  # embedding dimensions
        self.init_fn: Init | None = None

    def compile(self, optimizer: Optimizer | None = None) -> None:
        super().compile(optimizer)
        vocab_size = self.x.shape[1]

        # set initializer
        initializer_params = inits.InitParams(vocab_size, self.act_fn_name)
        self.init_fn = inits.INITS[self.init_fn_name](initializer_params)

        # init weights (vocab_size, c_out)
        self.w = self.init_fn((vocab_size, self.out_channels))
        self.parameters.append(self.w)

    def forward(self, mode: str = "eval") -> None:
        self.y.data = (self.x @ self.w).data

    def backward(self) -> None:
        self.w.grad = self.x.T @ self.y.grad


@dataclass(init=False, repr=False)
class Block(ParamLayer):
    """Context block embedding layer."""

    def __init__(
        self,
        out_channels: int,
        init: str = "kaiming_he",
        input_shape: ShapeLike | None = None,
    ) -> None:
        """Context embedding layer.

        Parameters
        ----------
        out_channels : int
            Number of output channels (embedding dimensions) of the layer.
        init : str, optional
            Initialization function for weights, by default "kaiming_he".
        input_shape : ShapeLike | None, optional
            Shape of a sample. Required if the layer is used as input, by default None.
        """
        super().__init__(
            init_fn_name=init,
            input_shape=input_shape,
        )
        self.out_channels = out_channels  # embedding dimensions
        self.init_fn: Init | None = None
        self.block_size: int = 0

    def compile(self, optimizer: Optimizer | None = None) -> None:
        super().compile(optimizer)
        vocab_size = self.x.shape[2]

        # set initializer
        initializer_params = inits.InitParams(vocab_size, self.act_fn_name)
        self.init_fn = inits.INITS[self.init_fn_name](initializer_params)

        # init weights (vocab_size, c_out)
        self.w = self.init_fn((vocab_size, self.out_channels))
        self.parameters.append(self.w)

    def forward(self, mode: str = "eval") -> None:
        y = self.x @ self.w
        # flatten trailing dims to make output 2 dimensonal
        self.y.data = y.reshape((self.x.shape[0], -1)).data  # (s, b*d)

    def backward(self) -> None:
        # TODO: simplify
        # (s, b, d) from (x, b*d)
        y_grad = self.y.grad.reshape((*self.x.shape[:2], self.out_channels))
        x_ma = np.moveaxis(self.x.data, 0, -1)  # (b, v, s)
        y_grad_ma = np.moveaxis(y_grad, 1, 0)  # (b, s, d)
        self.w.grad = np.sum(x_ma @ y_grad_ma, axis=0)
