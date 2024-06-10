"""Tensor module"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Optional

import numpy

from ._types import _ArrayLike, _DeviceLike, _DtypeLike, _ScalarLike, _ShapeLike
from .engine import (
    _check_device_availability,
    _cupy_to_numpy,
    _get_engine,
    _infer_device,
    _numpy_to_cupy,
)

__all__ = ["tensor", "Tensor"]


def _as_arraylike(value: Any) -> _ArrayLike:
    if isinstance(value, Tensor):
        return value.data
    return value


def _as_tensor(value: _ArrayLike | _ScalarLike) -> Tensor:
    if isinstance(value, _ArrayLike):
        return Tensor(value)
    return tensor(value)


def tensor(
    data: _ArrayLike | _ScalarLike,
    device: Optional[_DeviceLike] = None,
    dtype: Optional[_DtypeLike] = None,
    copy: bool = False,
    requires_grad: bool = True,
) -> Tensor:
    """Creates a tensor object.

    Parameters
    ----------
    data : _ArrayLike | _ScalarLike
        Data to initialize the tensor.
    device : _DeviceLike, optional
        Device the tensor should be stored on. If None it is inferred from the data.
    dtype : _DtypeLike, optional
        Data type of tensor data. If None it is inferred from the data.
    copy: bool, optional
        If true, the data object is copied (may impact performance), by default False.
    requires_grad: bool, optional
        Whether the tensor requires gradients, by default True.
    """
    device = _infer_device(data) if device is None else device
    data = _get_engine(device).array(data, copy=copy, dtype=dtype)
    return Tensor(data, requires_grad)


class Tensor:
    """Tensor object."""

    def __init__(
        self,
        data: _ArrayLike,
        requires_grad: bool = True,
    ) -> None:
        """Tensor object.

        Parameters
        ----------
        data : _ArrayLike
            Data to initialize the tensor.
        requires_grad: bool, optional
            Whether the tensor requires gradients, by default True.
        """
        self.data = data
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
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
            raise ValueError(
                f"Invalid data type {type(value)}. Use compyute.tensor to initialize tensors."
            )
        self._data = value

    @property
    def _engine(self) -> ModuleType:
        return _get_engine(self.device)

    @property
    def device(self) -> _DeviceLike:
        """Device the tensor is stored on."""
        return _infer_device(self._data)

    def to_device(self, device: _DeviceLike) -> None:
        """Moves the tensor to a specified device."""
        if self.device == device:
            return
        _check_device_availability(device)

        if device == "cuda":
            self._data = _numpy_to_cupy(self._data)
        else:
            self._data = _cupy_to_numpy(self._data)

        if self.grad is not None:
            self.grad.to_device(device)

    @property
    def dtype(self) -> _DtypeLike:
        """Tensor data type."""
        return str(self._data.dtype)

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
        """Transposed tensor."""
        return Tensor(self._engine.moveaxis(self._data, -2, -1))

    # ----------------------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        array_string = numpy.array2string(
            self.cpu().data,
            max_line_width=100,
            prefix="Tensor(",
            separator=", ",
            precision=4,
            floatmode="maxprec_equal",
        )
        return f"Tensor({array_string})"

    def __getitem__(self, key: Any) -> Tensor:
        i = tuple(_as_arraylike(j) for j in key) if isinstance(key, tuple) else _as_arraylike(key)
        return _as_tensor(self._data[i])

    def __setitem__(self, key: Any, value: Tensor | _ScalarLike) -> None:
        self._data[_as_arraylike(key)] = _as_arraylike(value)

    def __iter__(self) -> Tensor:
        self._iterator = 0
        return self

    def __next__(self) -> Tensor | _ScalarLike:
        if self._iterator < self.shape[0]:
            data = self[self._iterator]
            self._iterator += 1
            return data
        raise StopIteration

    def __add__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _as_tensor(self._data + _as_arraylike(other))

    def __radd__(self, other: _ScalarLike) -> Tensor:
        other = 0.0 if other is None else other  # for gradient accumulation
        return self + other

    def __mul__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _as_tensor(self._data * _as_arraylike(other))

    def __rmul__(self, other: _ScalarLike) -> Tensor:
        return self * other

    def __pow__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _as_tensor(self._data ** _as_arraylike(other))

    def __rpow__(self, other: _ScalarLike) -> Tensor:
        return tensor(other, self.device) ** self

    def __neg__(self) -> Tensor:
        return self * -1

    def __sub__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _as_tensor(self._data - _as_arraylike(other))

    def __rsub__(self, other: _ScalarLike) -> Tensor:
        return -self + other

    def __truediv__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _as_tensor(self._data / _as_arraylike(other))

    def __rtruediv__(self, other: _ScalarLike) -> Tensor:
        return self**-1 * other

    def __floordiv__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _as_tensor(self._data // _as_arraylike(other))

    def __rfloordiv__(self, other: _ScalarLike) -> Tensor:
        return (other // self).astype(self.dtype)

    def __mod__(self, other: int) -> Tensor:
        return _as_tensor(self._data % other)

    def __matmul__(self, other: Tensor) -> Tensor:
        return _as_tensor(self._data @ other.data)

    def __lt__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _as_tensor(self._data < _as_arraylike(other))

    def __gt__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _as_tensor(self._data > _as_arraylike(other))

    def __le__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _as_tensor(self._data <= _as_arraylike(other))

    def __ge__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _as_tensor(self._data >= _as_arraylike(other))

    def __eq__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _as_tensor(self._data == _as_arraylike(other))

    def __ne__(self, other: Tensor | _ScalarLike) -> Tensor:
        return _as_tensor(self._data != _as_arraylike(other))

    def __iadd__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data += _as_arraylike(other)
        return self

    def __isub__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data -= _as_arraylike(other)
        return self

    def __imul__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data *= _as_arraylike(other)
        return self

    def __idiv__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data /= _as_arraylike(other)
        return self

    def __ifloordiv__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data //= _as_arraylike(other)
        return self

    def __imod__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data %= _as_arraylike(other)
        return self

    def __ipow__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data **= _as_arraylike(other)
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

    def astype(self, dtype: _DtypeLike) -> Tensor:
        """Returns a copy of the tensor with parsed values.

        Parameters
        ----------
        dtype : _DtypeLike
            Datatype to convert the tensor to.

        Returns
        -------
        Tensor
            Tensor of dtype.
        """
        return tensor(self._data, self.device, dtype=dtype)

    def int(self) -> Tensor:
        """Returns a copy of the tensor with int values."""
        return self.astype("int32")

    def long(self) -> Tensor:
        """Returns a copy of the tensor with long values."""
        return self.astype("int64")

    def half(self) -> Tensor:
        """Returns a copy of the tensor with half precision values."""
        return self.astype("float16")

    def float(self) -> Tensor:
        """Returns a copy of the tensor with float values."""
        return self.astype("float32")

    def double(self) -> Tensor:
        """Returns a copy of the tensor with double precision values."""
        return self.astype("float64")

    def complex(self) -> Tensor:
        """Returns a copy of the tensor with complex values."""
        return self.astype("complex64")

    def to_numpy(self) -> numpy.ndarray:
        """Returns the tensor data as a Numpy array."""
        return self.cpu().data

    # ----------------------------------------------------------------------------------------------
    # MEMORY/DEVICE METHODS
    # ----------------------------------------------------------------------------------------------

    def copy(self) -> Tensor:
        """Returns a copy of the tensor."""
        t = Tensor(self._data.copy(), requires_grad=self.requires_grad)
        t.grad = None if self.grad is None else self.grad.copy()
        return t

    def item(self) -> _ScalarLike:
        """Returns the scalar value of the tensor data."""
        return self._data.item()

    def cpu(self):
        """Returns a copy of the tensor on the cpu."""
        if self.device == "cpu":
            return self

        new_tensor = self.copy()
        new_tensor.to_device("cpu")
        return new_tensor

    def cuda(self):
        """Returns a copy of the tensor on the gpu."""
        if self.device == "cuda":
            return self

        new_tensor = self.copy()
        new_tensor.to_device("cuda")
        return new_tensor
