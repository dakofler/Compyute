"""Base tensor class."""

from __future__ import annotations

from types import ModuleType
from typing import Any, Optional, TypeAlias

import numpy

from .dtypes import Dtype, _DtypeLike, _ScalarLike
from .engine import Device, _ArrayLike, _DeviceLike, data_to_device, get_array_string, get_engine, infer_device

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
) -> Tensor:
    """Creates a tensor object from arbitrary data.
    The data type and device are inferred from the data if not specified.

    Parameters
    ----------
    data : _ArrayLike | _ScalarLike
        Data to initialize the tensor.
    device : _DeviceLike, optional
        Device the tensor should be stored on. If ``None``, it is inferred from the data.
    dtype : _DtypeLike, optional
        Data type of tensor data. If ``None``, it is inferred from the data.
    copy : bool, optional
        If ``True``, the data object is copied (may impact performance, uses more memory).
        Defaults to ``False``.

    Returns
    -------
    Tensor
        Tensor object.
    """
    if isinstance(data, _ArrayLike) and device is None and dtype is None:
        return Tensor(data)

    device = infer_device(type(data)) if device is None else device
    dtype = Dtype(dtype).value if dtype is not None else None
    data_array = get_engine(device).array(data, dtype, copy=copy)

    return Tensor(data_array)


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
    """

    def __init__(self, data: _ArrayLike) -> None:
        self.data = data
        self.grad: Optional[Tensor] = None
        self._engine: Optional[ModuleType] = None
        self._device: Optional[Device] = None
        self._dtype: Optional[Dtype] = None
        self._iterator: int = 0

    # ----------------------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @property
    def data(self) -> _ArrayLike:
        """Tensor data."""
        return self._data

    @data.setter
    def data(self, value: _ArrayLike) -> None:
        if not isinstance(value, _ArrayLike):
            raise TypeError("Invalid data type. Use ``compyute.tensor()`` to initialize tensors.")
        self._data = value

    @property
    def device(self) -> Device:
        """Device the tensor data is stored on."""
        if self._device is None:
            self._device = infer_device(type(self._data))
        return self._device

    @property
    def dtype(self) -> Dtype:
        """Tensor data type."""
        if self._dtype is None:
            self._dtype = Dtype(str(self._data.dtype))
        return self._dtype

    @property
    def engine(self) -> ModuleType:
        """Computation engine."""
        return get_engine(self.device)

    @property
    def n_axes(self) -> int:
        """Number of tensor axes."""
        return self._data.ndim

    @property
    def size(self) -> int:
        """Tensor size (number of elements)."""
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
        """Transposed view of the tensor (last two axes)."""
        return Tensor(self.engine.moveaxis(self._data, -2, -1))

    @property
    def ptr(self) -> int:
        """Pointer to the tensor data."""
        return id(self._data)

    def _cleanup(self) -> None:
        """Resets chached information."""
        self._engine = None
        self._device = None
        self._dtype = None
        self.grad = None

    # ----------------------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        array_string = get_array_string(self.to_numpy())
        return f"Tensor({array_string})"

    def __getitem__(self, key: Any) -> Tensor:
        key = tuple(_to_arraylike(k) for k in key) if isinstance(key, tuple) else _to_arraylike(key)
        return tensor(self._data[key])

    def __setitem__(self, key: Any, value: Tensor | _ScalarLike) -> None:
        self._data[_to_arraylike(key)] = _to_arraylike(value)

    def __iter__(self) -> Tensor:
        self._iterator = 0
        return self

    def __next__(self) -> Tensor | _ScalarLike:
        if self._iterator < self.shape[0]:
            self._iterator += 1
            return self[self._iterator - 1]
        raise StopIteration

    def __add__(self, other: Tensor | _ScalarLike) -> Tensor:
        return tensor(self._data + _to_arraylike(other))

    def __radd__(self, other: Optional[_ScalarLike]) -> Tensor:
        other = other or 0.0  # for gradient accumulation
        return self + other

    def __mul__(self, other: Tensor | _ScalarLike) -> Tensor:
        return tensor(self._data * _to_arraylike(other))

    def __rmul__(self, other: _ScalarLike) -> Tensor:
        return self * other

    def __pow__(self, other: Tensor | _ScalarLike) -> Tensor:
        return tensor(self._data ** _to_arraylike(other))

    def __rpow__(self, other: _ScalarLike) -> Tensor:
        return tensor(other, self.device) ** self

    def __neg__(self) -> Tensor:
        return self * -1

    def __sub__(self, other: Tensor | _ScalarLike) -> Tensor:
        return tensor(self._data - _to_arraylike(other))

    def __rsub__(self, other: _ScalarLike) -> Tensor:
        return -self + other

    def __truediv__(self, other: Tensor | _ScalarLike) -> Tensor:
        return tensor(self._data / _to_arraylike(other))

    def __rtruediv__(self, other: _ScalarLike) -> Tensor:
        return self**-1 * other

    def __floordiv__(self, other: Tensor | _ScalarLike) -> Tensor:
        return tensor(self._data // _to_arraylike(other))

    def __rfloordiv__(self, other: _ScalarLike) -> Tensor:
        return (other // self).to_type(self.dtype)

    def __mod__(self, other: int) -> Tensor:
        return tensor(self._data % other)

    def __matmul__(self, other: Tensor) -> Tensor:
        return tensor(self._data @ other.data)

    def __lt__(self, other: Tensor | _ScalarLike) -> Tensor:
        return tensor(self._data < _to_arraylike(other))

    def __gt__(self, other: Tensor | _ScalarLike) -> Tensor:
        return tensor(self._data > _to_arraylike(other))

    def __le__(self, other: Tensor | _ScalarLike) -> Tensor:
        return tensor(self._data <= _to_arraylike(other))

    def __ge__(self, other: Tensor | _ScalarLike) -> Tensor:
        return tensor(self._data >= _to_arraylike(other))

    def __eq__(self, other: Tensor | _ScalarLike) -> Tensor:
        return tensor(self._data == _to_arraylike(other))

    def __ne__(self, other: Tensor | _ScalarLike) -> Tensor:
        return tensor(self._data != _to_arraylike(other))

    def __iadd__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data += _to_arraylike(other)
        return self

    def __isub__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data -= _to_arraylike(other)
        return self

    def __imul__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data *= _to_arraylike(other)
        return self

    def __idiv__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data /= _to_arraylike(other)
        return self

    def __ifloordiv__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data //= _to_arraylike(other)
        return self

    def __imod__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data %= _to_arraylike(other)
        return self

    def __ipow__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data **= _to_arraylike(other)
        return self

    def __len__(self) -> int:
        return self.shape[0]

    def __hash__(self):
        return id(self)

    def __array__(self, dtype=None, copy=None):
        return self.to_numpy()

    # ----------------------------------------------------------------------------------------------
    # DEVICE CONVERSIONS
    # ----------------------------------------------------------------------------------------------

    def to_device(self, device: _DeviceLike) -> Tensor:
        """Returns a copy of the tensor on the specified device.

        Parameters
        ----------
        device : _DeviceLike
            Device to move the tensor to.

        Returns
        -------
        Tensor
            Tensor on the specified device.
        """
        device = Device(device)
        if self._device == device:
            return self

        new_data = data_to_device(self._data, device)
        new_tensor = Tensor(new_data)
        if self.grad is not None:
            new_tensor.grad = self.grad.to_device(device)
        return new_tensor

    def _to_device(self, device: _DeviceLike) -> None:
        """Moves the tensor to the specified device.

        Parameters
        ----------
        device : _DeviceLike
            Device to move the tensor to.
        """
        device = Device(device)
        if self._device == device:
            return

        self._device = device
        self._engine = get_engine(self.device)
        self.data = data_to_device(self._data, device)
        if self.grad is not None:
            self.grad = self.grad.to_device(device)

    # ----------------------------------------------------------------------------------------------
    # DTYPE CONVERSIONS
    # ----------------------------------------------------------------------------------------------

    def to_type(self, dtype: _DtypeLike) -> Tensor:
        """Returns a copy of the tensor with elements cast to the given dtype.

        Parameters
        ----------
        dtype : _DtypeLike
            Datatype to cast tensor-elements to.

        Returns
        -------
        Tensor
            Tensor with elements cast to the given dtype.
        """
        dtype = Dtype(dtype)
        if self.dtype == dtype:
            return self

        return Tensor(self._data.astype(dtype.value))

    def _to_type(self, dtype: _DtypeLike) -> None:
        """Casts tensor elements to the given dtype.

        Parameters
        ----------
        dtype : _DtypeLike
            Datatype to cast tensor-elements to.
        """
        dtype = Dtype(dtype)
        if self.dtype == dtype:
            return

        self._dtype = dtype
        self.data = self._data.astype(dtype.value)

    def to_int(self) -> Tensor:
        """Returns a copy of the tensor with integer values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.int32` values.
        """
        return self.to_type(Dtype.INT32)

    def to_long(self) -> Tensor:
        """Returns a copy of the tensor with long integer values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.int64` values.
        """
        return self.to_type(Dtype.INT64)

    def to_half(self) -> Tensor:
        """Returns a copy of the tensor with half precision values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.float16` values.
        """
        return self.to_type(Dtype.FLOAT16)

    def to_float(self) -> Tensor:
        """Returns a copy of the tensor with single precision values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.float32` values.
        """
        return self.to_type(Dtype.FLOAT32)

    def to_double(self) -> Tensor:
        """Returns a copy of the tensor with double precision values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.float64` values.
        """
        return self.to_type(Dtype.FLOAT64)

    def to_complex(self) -> Tensor:
        """Returns a copy of the tensor with complex values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.complex64` values.
        """
        return self.to_type(Dtype.COMPLEX64)

    # ----------------------------------------------------------------------------------------------
    # MEMORY/DEVICE METHODS
    # ----------------------------------------------------------------------------------------------

    def copy(self) -> Tensor:
        """Returns a copy of the tensor.

        Returns
        -------
        Tensor
            Copy of the tensor.
        """
        new_tensor = Tensor(self._data.copy())
        if self.grad is not None:
            new_tensor.grad = self.grad.copy()
        return new_tensor

    def item(self) -> _ScalarLike:
        """Returns the scalar value of the tensor data.
        Only works for scalar tensors.

        Returns
        -------
        _ScalarLike
            Scalar value of the tensor data.
        """
        return self._data.item()

    def to_cpu(self) -> Tensor:
        """Returns a copy of the tensor on the CPU.

        Returns
        -------
        Tensor
            Tensor on the CPU.
        """
        return self.to_device(Device.CPU)

    def to_cuda(self) -> Tensor:
        """Returns a copy of the tensor on the GPU.

        Returns
        -------
        Tensor
            Tensor on the GPU.
        """
        return self.to_device(Device.CUDA)

    # ----------------------------------------------------------------------------------------------
    # OTHER METHODS
    # ----------------------------------------------------------------------------------------------

    def to_numpy(self) -> numpy.ndarray:
        """Returns the tensor data as a NumPy array.

        Returns
        -------
        numpy.ndarray
            NumPy array of the tensor data.
        """
        return self.to_cpu().data

    def to_shape(self, shape: _ShapeLike) -> Tensor:
        """Returns a view of the tensor of a given shape.

        Parameters
        ----------
        shape : _ShapeLike
            Shape of the view.

        Returns
        -------
        Tensor
            View of the tensor.
        """
        return Tensor(self._data.reshape(shape))

    def to_list(self) -> list:
        """Returns the tensor data as a list.

        Returns
        -------
        list
            List of the tensor data.
        """
        return self._data.tolist()

    def masked_fill(self, mask: Tensor, value: _ScalarLike) -> None:
        """Sets elements of the tensor to a value, where the mask is true.

        Parameters
        ----------
        mask : Tensor
            Boolean mask tensor.
        value : _ScalarLike
            Value to set tensor elements to.
        """
        if mask.dtype.value != "bool":
            raise AttributeError("Mask must be a boolean tensor.")
        self.engine.putmask(self._data, mask.data, value)


def _to_arraylike(value: Any) -> _ArrayLike | _ScalarLike:
    """Converts a value to an array."""
    if isinstance(value, Tensor):
        return value.data
    return value
