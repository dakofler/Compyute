"""Base tensor class."""

from __future__ import annotations

from typing import Any, Optional, TypeAlias

import numpy

from .backend import (
    ArrayLike,
    Device,
    cpu,
    cuda,
    data_to_device,
    get_default_device,
    get_device_from_class,
)
from .typing import (
    DType,
    ScalarLike,
    complex64,
    float16,
    float32,
    float64,
    get_default_dtype,
    int32,
    int64,
)

__all__ = ["tensor", "Tensor"]

ShapeLike: TypeAlias = tuple[int, ...]
AxisLike: TypeAlias = int | tuple[int, ...]


class ShapeError(Exception):
    """Incompatible tensor shapes."""


def tensor(
    data: ArrayLike | ScalarLike,
    device: Optional[Device] = None,
    dtype: Optional[DType] = None,
) -> Tensor:
    """Creates a tensor object from arbitrary data.
    The data type and device are inferred from the data if not specified.

    Parameters
    ----------
    data : ArrayLike | _ScalarLike
        Data to initialize the tensor data.
        Can be a list, tuple, NumPy/Cupy ndarray, scalar, and other types.
    device : Device, optional
        Device the tensor should be stored on. If ``None``, it is inferred from the data.
    dtype : DType, optional
        Data type of tensor data. If ``None``, it is inferred from the data.

    Returns
    -------
    Tensor
        Tensor object.
    """
    device = device or get_default_device() or get_device_from_class(type(data))
    dtype = dtype or get_default_dtype()
    dtype_str = dtype.value if dtype is not None else None
    data_array = device.engine.asarray(data, dtype_str)

    return Tensor(data_array)


class Tensor:
    """Tensor object used for storing multidimensional data.

    .. note::
        Tensors can only be initialized with NumPy or CuPy arrays.
        For other data types use the :func:`compyute.tensor` function. It automatically
        infers the data type and device if not specified.

    Parameters
    ----------
    data : ArrayLike
        Data to initialize the tensor. Must be a NumPy array or CuPy array.
    """

    grad: Optional[Tensor] = None
    _iterator: int = 0

    def __init__(self, data: ArrayLike) -> None:
        self.data = data

    # ----------------------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @property
    def device(self) -> Device:
        """Device the tensor data is stored on."""
        return get_device_from_class(type(self.data))

    @property
    def dtype(self) -> DType:
        """Tensor data type."""
        return DType(str(self.data.dtype))

    @property
    def n_axes(self) -> int:
        """Number of tensor axes."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Tensor size (number of elements)."""
        return self.data.size

    @property
    def shape(self) -> ShapeLike:
        """Tensor shape."""
        return self.data.shape

    @property
    def strides(self) -> tuple[int, ...]:
        """Tensor strides."""
        return self.data.strides

    @property
    def T(self) -> Tensor:
        """View of the tensor with its last two axes transposed."""
        return Tensor(self.data.transpose(*range(self.n_axes - 2), -1, -2))

    @property
    def ptr(self) -> int:
        """Pointer to the tensor data in memory."""
        return id(self.data)

    # ----------------------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------------------

    def __repr__(self):
        return (
            "Tensor("
            + self.device.engine.array2string(
                self.data,
                100,
                4,
                separator=", ",
                prefix="Tensor(",
                floatmode="maxprec_equal",
            )
            + ")"
        )

    def __getitem__(self, key: Any) -> Tensor:
        if isinstance(key, tuple):
            return Tensor(self.data[key])
        key = to_arraylike(key)
        return Tensor(self.data[key])

    def __setitem__(self, key: Any, value: Tensor | ScalarLike) -> None:
        self.data[to_arraylike(key)] = to_arraylike(value)

    def __iter__(self) -> Tensor:
        self._iterator = 0
        return self

    def __next__(self) -> Tensor | ScalarLike:
        if self._iterator < self.shape[0]:
            self._iterator += 1
            return self[self._iterator - 1]
        raise StopIteration

    def __add__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data + to_arraylike(other))

    def __radd__(self, other: Optional[ScalarLike]) -> Tensor:
        # for gradient accumulation make None += Tensor to be 0 += Tensor
        return Tensor(self.data + (other or 0.0))

    def __iadd__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data += to_arraylike(other)
        return self

    def __sub__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data - to_arraylike(other))

    def __rsub__(self, other: ScalarLike) -> Tensor:
        return Tensor(other - self.data)

    def __isub__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data -= to_arraylike(other)
        return self

    def __mul__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data * to_arraylike(other))

    def __rmul__(self, other: ScalarLike) -> Tensor:
        return Tensor(other * self.data)

    def __imul__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data *= to_arraylike(other)
        return self

    def __truediv__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data / to_arraylike(other))

    def __rtruediv__(self, other: ScalarLike) -> Tensor:
        return Tensor(other / self.data)

    def __idiv__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data /= to_arraylike(other)
        return self

    def __floordiv__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data // to_arraylike(other))

    def __rfloordiv__(self, other: ScalarLike) -> Tensor:
        return Tensor(other // self.data)

    def __ifloordiv__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data //= to_arraylike(other)
        return self

    def __pow__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data ** to_arraylike(other))

    def __rpow__(self, other: ScalarLike) -> Tensor:
        return Tensor(other**self.data)

    def __ipow__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data **= to_arraylike(other)
        return self

    def __mod__(self, other: int) -> Tensor:
        return Tensor(self.data % other)

    def __rmod__(self, other: int) -> Tensor:
        return Tensor(other % self.data)

    def __imod__(self, other: int) -> Tensor:
        self.data %= other
        return self

    def __neg__(self) -> Tensor:
        return Tensor(-self.data)

    def __matmul__(self, other: Tensor) -> Tensor:
        return Tensor(self.data @ other.data)

    def __lt__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data < to_arraylike(other))

    def __gt__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data > to_arraylike(other))

    def __le__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data <= to_arraylike(other))

    def __ge__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data >= to_arraylike(other))

    def __eq__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data == to_arraylike(other))

    def __ne__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data != to_arraylike(other))

    def __len__(self) -> int:
        return self.shape[0]

    def __hash__(self):
        return id(self)

    def __array__(self, dtype=None, copy=None):
        return self.to_numpy()

    def __bool__(self) -> bool:
        return True

    # ----------------------------------------------------------------------------------------------
    # DEVICE CONVERSIONS
    # ----------------------------------------------------------------------------------------------

    def to_device(self, device: Device) -> Tensor:
        """Returns a copy of the tensor on the specified device.

        Parameters
        ----------
        device : Device
            Device to move the tensor to.

        Returns
        -------
        Tensor
            Tensor on the specified device.
        """
        if self.device == device:
            return self

        new_tensor = Tensor(data_to_device(self.data, device))
        if self.grad:
            new_tensor.grad = self.grad.to_device(device)
        return new_tensor

    def ito_device(self, device: Device) -> None:
        """Inplace operation to move the tensor to the specified device.

        Parameters
        ----------
        device : Device
            Device to move the tensor to.
        """
        if self.device == device:
            return

        self.data = data_to_device(self.data, device)
        if self.grad:
            self.grad.ito_device(device)

    def to_cpu(self) -> Tensor:
        """Returns a copy of the tensor on the CPU.

        Returns
        -------
        Tensor
            Tensor on the CPU.
        """
        return self.to_device(cpu)

    def to_cuda(self) -> Tensor:
        """Returns a copy of the tensor on the GPU.

        Returns
        -------
        Tensor
            Tensor on the GPU.
        """
        return self.to_device(cuda)

    # ----------------------------------------------------------------------------------------------
    # DTYPE CONVERSIONS
    # ----------------------------------------------------------------------------------------------

    def to_type(self, dtype: DType) -> Tensor:
        """Returns a copy of the tensor with elements cast to the given dtype.

        Parameters
        ----------
        dtype : DType
            Datatype to cast tensor-elements to.

        Returns
        -------
        Tensor
            Tensor with elements cast to the given dtype.
        """
        if self.dtype == dtype:
            return self

        return Tensor(self.data.astype(dtype.value))

    def to_int(self) -> Tensor:
        """Returns a copy of the tensor with integer values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.int32` values.
        """
        return self.to_type(int32)

    def to_long(self) -> Tensor:
        """Returns a copy of the tensor with long integer values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.int64` values.
        """
        return self.to_type(int64)

    def to_half(self) -> Tensor:
        """Returns a copy of the tensor with half precision values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.float16` values.
        """
        return self.to_type(float16)

    def to_float(self) -> Tensor:
        """Returns a copy of the tensor with single precision values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.float32` values.
        """
        return self.to_type(float32)

    def to_double(self) -> Tensor:
        """Returns a copy of the tensor with double precision values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.float64` values.
        """
        return self.to_type(float64)

    def to_complex(self) -> Tensor:
        """Returns a copy of the tensor with complex values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.complex64` values.
        """
        return self.to_type(complex64)

    # ----------------------------------------------------------------------------------------------
    # OTHER METHODS
    # ----------------------------------------------------------------------------------------------

    def copy(self) -> Tensor:
        """Returns a copy of the tensor.

        Returns
        -------
        Tensor
            Copy of the tensor.
        """
        new_tensor = Tensor(self.data.copy())
        if self.grad:
            new_tensor.grad = self.grad.copy()
        return new_tensor

    def item(self) -> ScalarLike:
        """Returns the scalar value of the tensor data.
        Only works for scalar tensors.

        Returns
        -------
        _ScalarLike
            Scalar value of the tensor data.
        """
        return self.data.item()

    def to_numpy(self) -> numpy.ndarray:
        """Returns the tensor data as a NumPy array.

        Returns
        -------
        numpy.ndarray
            NumPy array of the tensor data.
        """
        return self.to_cpu().data

    def to_shape(self, shape: ShapeLike) -> Tensor:
        """Returns a view of the tensor of a given shape.

        Parameters
        ----------
        shape : ShapeLike
            Shape of the view.

        Returns
        -------
        Tensor
            View of the tensor.
        """
        return Tensor(self.data.reshape(shape))

    def to_list(self) -> list:
        """Returns the tensor data as a list.

        Returns
        -------
        list
            List of the tensor data.
        """
        return self.data.tolist()


def to_arraylike(value: Any) -> ArrayLike | ScalarLike:
    """Converts a value to an array like."""
    if isinstance(value, Tensor):
        return value.data
    return value
