"""Neural network parameter and buffer classes."""

from __future__ import annotations

from ..base_tensor import Tensor, _ArrayLike
from ..dtypes import FLOAT_DTYPES
from ..engine import Device, _DeviceLike, data_to_device

__all__ = ["Buffer", "Parameter"]


class Parameter(Tensor):
    """Trainable neural network parameter.

    Parameters
    ----------
    data : Tensor | _ArrayLike
        Data to initialize the parameter. Must be of type ``float``.

    Raises
    ------
    TypeError
        If an invalid data type is provided.
    """

    def __init__(self, data: Tensor | _ArrayLike) -> None:
        if isinstance(data, Tensor):
            data = data.data
        if str(data.dtype) not in FLOAT_DTYPES:
            raise TypeError("Invalid data type for parameter. Must be float.")
        super().__init__(data)

    def to_device(self, device: _DeviceLike) -> Parameter:
        """Returns a copy of the parameter on the specified device.

        Parameters
        ----------
        device : _DeviceLike
            Device to move the parameter to.

        Returns
        -------
        Parameter
            Parameter on the specified device.
        """
        device = Device(device)
        if self._device == device:
            return self

        new_data = data_to_device(self._data, device)
        new_param = Parameter(new_data)
        if self.grad is not None:
            new_param.grad = self.grad.to_device(device)
        return new_param


class Buffer(Tensor):
    """Neural network buffer variable.

    Parameters
    ----------
    data : Tensor | _ArrayLike
        Data to initialize the buffer.
    """

    def __init__(self, data: Tensor | _ArrayLike) -> None:
        if isinstance(data, Tensor):
            data = data.data
        super().__init__(data)

    def to_device(self, device: _DeviceLike) -> Buffer:
        """Returns a copy of the buffer on the specified device.

        Parameters
        ----------
        device : _DeviceLike
            Device to move the buffer to.

        Returns
        -------
        Buffer
            Buffer on the specified device.
        """
        device = Device(device)
        if self._device == device:
            return self

        new_data = data_to_device(self._data, device)
        return Buffer(new_data)
