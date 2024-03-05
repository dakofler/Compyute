"""Neural network blocks module"""

from compyute.nn.containers import Sequential, ParallelAdd
from compyute.nn.layers import Linear, RecurrentCell
from compyute.nn.module import Module


__all__ = ["Recurrent", "Residual"]


class Recurrent(Sequential):
    """Recurrent neural network block."""

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        num_layers: int = 1,
        use_bias: bool = True,
        dtype: str = "float32",
    ) -> None:
        """Recurrent neural network block.

        Parameters
        ----------
        in_channels : int
            Number of input features.
        h_channels: int
            Number of hidden features used in each recurrent cell.
        num_layers: int, optional
            Number of recurrent layers in the block, by default 1.
        use_bias : bool, optional
            Whether to use bias values, by default True.
        """
        m = [Linear(in_channels, h_channels, use_bias=use_bias, dtype=dtype)]
        for i in range(num_layers):
            if i > 0:
                m.append(Linear(h_channels, h_channels, use_bias=use_bias, dtype=dtype))
            m.append(RecurrentCell(h_channels, use_bias=use_bias, dtype=dtype))
        super().__init__(m)


class Residual(ParallelAdd):
    """Block with residual connection."""

    def __init__(self, core_module: Module) -> None:
        """Block with residual connection bypassing the core module.

        Parameters
        ----------
        core_module : Module
            Core module bypassed by the residual connection. For multiple modules use a container as core module.
            To ensure matching tensor shapes, you might need to use a projection layer.
        """
        super().__init__([core_module, Module()])  # emtpy module as residual connection
