"""Base tensor class."""

from __future__ import annotations

from typing import Any, Optional, TypeAlias

import numpy

from .dtypes import Dtype, _DtypeLike, _ScalarLike, select_dtype_str
from .engine import (
    Device,
    _ArrayLike,
    _DeviceLike,
    get_array_string,
    get_engine,
    infer_device,
    move_data_to_device,
    select_device_or_infer,
)

__all__ = ["tensor", "Tensor"]

_ShapeLike: TypeAlias = tuple[int, ...]
_AxisLike: TypeAlias = int | tuple[int, ...]


class ShapeError(Exception):
    """Incompatible tensor shapes."""


def tensor(
    data: _ArrayLike | _ScalarLike,
    device: Optional[_DeviceLike] = None,
    dtype: Optional[_DtypeLike] = None,
    copy: bool = False,
    requires_grad: bool = True,
) -> Tensor:
    """Creates a tensor object from arbitrary data.
    The data type and device are inferred from the data if not specified.

    Parameters
    ----------
    data : _ArrayLike | _ScalarLike
        Data to initialize the tensor.
    device : _DeviceLike, optional
        Device the tensor should be stored on. If ``None``, the default device is used.
        If no default device is set, it is inferred from the data.
    dtype : _DtypeLike, optional
        Data type of tensor data. If ``None``, the default data type is used.
        If no default data type is set, it is inferred from the data.
    copy : bool, optional
        If true, the data object is copied (may impact performance). Defaults to ``False``.
    requires_grad : bool, optional
        Whether the tensor requires gradients. Defaults to ``True``.
        If false gradients are not computed within neural network modules for this tensor.

    Returns
    -------
    Tensor
        The initialized tensor.
    """
    if isinstance(data, _ArrayLike):
        return Tensor(data)

    device = select_device_or_infer(data, device)
    dtype_str = select_dtype_str(dtype)
    data = get_engine(device).array(data, copy=copy, dtype=dtype_str)

    return Tensor(data, requires_grad)


def _tensor(data: _ArrayLike | _ScalarLike) -> Tensor:
    if isinstance(data, _ArrayLike):
        return Tensor(data)

    device = infer_device(data)
    data = get_engine(device).array(data, copy=False)

    return Tensor(data)


class Tensor:
    """Tensor object used for storing multidimensional data.

    .. note::
        Tensors can only be initialized with NumPy or CuPy arrays.
        For other data types use the :func:`tensor` function. It automatically
        infers the data type and device if not specified.

    Parameters
    ----------
    data : _ArrayLike
        Data to initialize the tensor. Must be a NumPy array or CuPy array.
        for other data use the :func:`tensor` function.
    requires_grad : bool, optional
        Whether the tensor requires gradients. Defaults to ``True``.
        If false gradients are not computed within neural network modules for this tensor.
    """

    def __init__(
        self,
        data: _ArrayLike,
        requires_grad: bool = True,
    ) -> None:
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._iterator: int = 0

    # ----------------------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _as_array(value: Any) -> _ArrayLike:
        """Converts a value to an array."""
        if isinstance(value, Tensor):
            return value.data
        return value

    @property
    def data(self) -> _ArrayLike:
        """Tensor data."""
        return self._data

    @data.setter
    def data(self, value: _ArrayLike) -> None:
        if not isinstance(value, _ArrayLike):
            raise ValueError(
                f"Invalid data type {type(value)}. Use ``compyute.tensor()`` to initialize tensors."
            )
        self._data = value

    @property
    def grad(self) -> Optional[Tensor]:
        """Tensor data gradients."""
        return self._grad

    @grad.setter
    def grad(self, value: Optional[Tensor]) -> None:
        if value is not None and not self.requires_grad:
            raise ValueError("Gradients are not required for this tensor.")
        self._grad = value

    @property
    def device(self) -> Device:
        """Device the tensor is stored on."""
        return infer_device(self._data)

    def to_device(self, device: _DeviceLike) -> None:
        """Moves the tensor to a specified device.

        Parameters
        ----------
        device : _DeviceLike
            Device to move the tensor to.
        """
        self._data = move_data_to_device(self._data, Device(device))
        if self.grad is not None:
            self.grad.to_device(device)

    @property
    def dtype(self) -> Dtype:
        """Tensor data type."""
        return Dtype(str(self._data.dtype))

    @property
    def ndim(self) -> int:
        """Number of tensor dimensions."""
        return self._data.ndim

    @property
    def size(self) -> int:
        """Tensor size."""
        return self._data.size

    @property
    def shape(self) -> _ShapeLike:
        """Tensor shape."""
        return self._data.shape

    @property
    def strides(self) -> tuple[int, ...]:
        """Tensor strides."""
        return self._data.strides

    @property
    def T(self) -> Tensor:
        """Returns a transposed version of the tensor."""
        return Tensor(get_engine(self.device).moveaxis(self._data, -2, -1))

    # ----------------------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        array_string = get_array_string(self.to_numpy())
        return f"Tensor({array_string})"

    def __getitem__(self, key: Any) -> Tensor:
        i = tuple(self._as_array(j) for j in key) if isinstance(key, tuple) else self._as_array(key)
        return tensor(self._data[i], self.device, self.dtype)

    def __setitem__(self, key: Any, value: Tensor | _ScalarLike) -> None:
        self._data[self._as_array(key)] = self._as_array(value)

    def __iter__(self) -> Tensor:
        self._iterator = 0
        return self

    def __next__(self) -> Tensor | _ScalarLike:
        if self._iterator < self.shape[0]:
            self._iterator += 1
            return self[self._iterator - 1]
        raise StopIteration

    def __add__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _tensor(self._data + self._as_array(other))

    def __radd__(self, other: Optional[_ScalarLike]) -> Tensor:
        other = other or 0.0  # for gradient accumulation
        return self + other

    def __mul__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _tensor(self._data * self._as_array(other))

    def __rmul__(self, other: _ScalarLike) -> Tensor:
        return self * other

    def __pow__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _tensor(self._data ** self._as_array(other))

    def __rpow__(self, other: _ScalarLike) -> Tensor:
        return tensor(other, self.device, self.dtype) ** self

    def __neg__(self) -> Tensor:
        return self * -1

    def __sub__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _tensor(self._data - self._as_array(other))

    def __rsub__(self, other: _ScalarLike) -> Tensor:
        return -self + other

    def __truediv__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _tensor(self._data / self._as_array(other))

    def __rtruediv__(self, other: _ScalarLike) -> Tensor:
        return self**-1 * other

    def __floordiv__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _tensor(self._data // self._as_array(other))

    def __rfloordiv__(self, other: _ScalarLike) -> Tensor:
        return (other // self).as_type(self.dtype)

    def __mod__(self, other: int) -> Tensor:
        return _tensor(self._data % other)

    def __matmul__(self, other: Tensor) -> Tensor:
        return _tensor(self._data @ other.data)

    def __lt__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _tensor(self._data < self._as_array(other))

    def __gt__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _tensor(self._data > self._as_array(other))

    def __le__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _tensor(self._data <= self._as_array(other))

    def __ge__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _tensor(self._data >= self._as_array(other))

    def __eq__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _tensor(self._data == self._as_array(other))

    def __ne__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _tensor(self._data != self._as_array(other))

    def __iadd__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data += self._as_array(other)
        return self

    def __isub__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data -= self._as_array(other)
        return self

    def __imul__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data *= self._as_array(other)
        return self

    def __idiv__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data /= self._as_array(other)
        return self

    def __ifloordiv__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data //= self._as_array(other)
        return self

    def __imod__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data %= self._as_array(other)
        return self

    def __ipow__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data **= self._as_array(other)
        return self

    def __len__(self) -> int:
        return self.shape[0]

    def __hash__(self):
        return id(self)

    def __array__(self, dtype=None, copy=None):
        return self.to_numpy()

    # ----------------------------------------------------------------------------------------------
    # DTYPE CONVERSIONS
    # ----------------------------------------------------------------------------------------------

    def as_type(self, dtype: _DtypeLike) -> Tensor:
        """Returns a copy of the tensor of a given data type.

        Parameters
        ----------
        dtype : _DtypeLike
            Datatype to convert the tensor to.

        Returns
        -------
        Tensor
            Tensor of a given data type.
        """
        dtype = Dtype(dtype)
        if dtype == self.dtype:
            return self

        new_tensor = Tensor(self._data.astype(dtype.value))
        if self.grad is not None:
            new_tensor.grad = self.grad.as_type(dtype)
        return new_tensor

    def int(self) -> Tensor:
        """Returns a copy of the tensor with data type :class:`compyute.int32`.

        Returns
        -------
        Tensor
            Tensor with data type :class:`compyute.int32`.
        """
        return self.as_type(Dtype.INT32)

    def long(self) -> Tensor:
        """Returns a copy of the tensor with data type :class:`compyute.int64`.

        Returns
        -------
        Tensor
            Tensor with data type :class:`compyute.int64`.
        """
        return self.as_type(Dtype.INT64)

    def half(self) -> Tensor:
        """Returns a copy of the tensor with data type :class:`compyute.float16`.

        Returns
        -------
        Tensor
            Tensor with data type :class:`compyute.float16`.
        """
        return self.as_type(Dtype.FLOAT16)

    def float(self) -> Tensor:
        """Returns a copy of the tensor with data type :class:`compyute.float32`.

        Returns
        -------
        Tensor
            Tensor with data type :class:`compyute.float32`.
        """
        return self.as_type(Dtype.FLOAT32)

    def double(self) -> Tensor:
        """Returns a copy of the tensor with data type :class:`compyute.float64`.

        Returns
        -------
        Tensor
            Tensor with data type :class:`compyute.float64`.
        """
        return self.as_type(Dtype.FLOAT64)

    def complex(self) -> Tensor:
        """Returns a copy of the tensor with data type :class:`compyute.complex64`.

        Returns
        -------
        Tensor
            Tensor with data type :class:`compyute.complex64`.
        """
        return self.as_type(Dtype.COMPLEX64)

    # ----------------------------------------------------------------------------------------------
    # MEMORY/DEVICE METHODS
    # ----------------------------------------------------------------------------------------------

    def copy(self) -> Tensor:
        """Returns a copy of the tensor."""
        new_tensor = Tensor(self._data.copy(), requires_grad=self.requires_grad)
        new_tensor.grad = None if self.grad is None else self.grad.copy()
        return new_tensor

    def item(self) -> _ScalarLike:
        """Returns the scalar value of the tensor data."""
        return self._data.item()

    def cpu(self):
        """Returns a copy of the tensor on the cpu."""
        if self.device == Device.CPU:
            return self
        new_tensor = self.copy()
        new_tensor.to_device(Device.CPU)
        return new_tensor

    def cuda(self):
        """Returns a copy of the tensor on the gpu."""
        if self.device == Device.CUDA:
            return self
        new_tensor = self.copy()
        new_tensor.to_device(Device.CUDA)
        return new_tensor

    # ----------------------------------------------------------------------------------------------
    # OTHER METHODS
    # ----------------------------------------------------------------------------------------------

    def to_numpy(self) -> numpy.ndarray:
        """Returns the tensor data as a NumPy array."""
        return self.cpu().data

    def as_shape(self, shape: _ShapeLike) -> Tensor:
        """Returns a view of the tensor of a given shape."""
        return Tensor(self._data.reshape(shape))
