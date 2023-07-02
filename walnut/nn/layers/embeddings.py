"""Embedding layers module"""

from dataclasses import dataclass

from walnut.tensor import ShapeLike
from walnut.nn.optimizers import Optimizer
from walnut.nn import inits
from walnut.nn.inits import Init
from walnut.nn.layers.parameter import ParamLayer


@dataclass(init=False, repr=False)
class Character(ParamLayer):
    """Fully connected layer."""

    def __init__(
        self,
        out_channels: int,
        init: str = "kaiming_he",
        input_shape: ShapeLike | None = None,
    ) -> None:
        """Character level embedding layer.

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
        in_channels = self.x.shape[1]  # vocab size

        # set initializer
        initializer_params = inits.InitParams(in_channels, self.act_fn_name)
        self.init_fn = inits.INITS[self.init_fn_name](initializer_params)

        # init weights (c_in, c_out)
        self.w = self.init_fn((in_channels, self.out_channels))
        self.parameters.append(self.w)

    def forward(self, mode: str = "eval") -> None:
        self.y.data = (self.x @ self.w).data

    def backward(self) -> None:
        self.w.grad = self.x.T @ self.y.grad
