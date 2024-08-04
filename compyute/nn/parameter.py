"""Neural network parameter and buffer classes."""

from ..base_tensor import Tensor

__all__ = ["Buffer", "Parameter"]


class Parameter(Tensor):
    """Trainable neural network parameter.

    Parameters
    ----------
    data : Tensor
        Data to initialize the parameter.
    requires_grad : bool, optional
        Whether the parameter requires gradients. Defaults to ``True``.
        If ``False`` gradients are not computed within neural network modules for this parameter.
    """

    def __init__(self, data: Tensor, requires_grad: bool = True) -> None:
        super().__init__(data.data, requires_grad)


class Buffer(Tensor):
    """Non-trainable neural network buffer variable.

    Parameters
    ----------
    data : Tensor
        Data to initialize the buffer.
    """

    def __init__(self, data: Tensor) -> None:
        super().__init__(data.data, requires_grad=False)
