"""Tensor module"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Optional, Sequence

import numpy

from ._types import (
    _ArrayLike,
    _AxisLike,
    _ComplexLike,
    _DeviceLike,
    _DtypeLike,
    _ScalarLike,
    _ShapeLike,
)
from .engine import (
    _check_device_availability,
    _cupy_to_numpy,
    _get_engine,
    _infer_device,
    _numpy_to_cupy,
)

__all__ = ["tensor", "Tensor"]


class ShapeError(Exception):
    """Incompatible tensor shapes."""


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
    data : ArrayLike | ScalarLike
        Data to initialize the tensor.
    device : DeviceLike, optional
        Device the tensor should be stored on. If None it is inferred from the data.
    dtype : DtypeLike, optional
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
        data : ArrayLike
            Data to initialize the tensor.
        requires_grad: bool, optional
            Whether the tensor requires gradients, by default True.
        """
        self.data = data
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
        self.__iterator: int = 0

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
        """Tensor datatype."""
        return str(self._data.dtype)

    @property
    def ndim(self) -> int:
        """Tensor dimensions."""
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
        """Tensor transposed."""
        return self.transpose()

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
        i = (
            tuple(self._as_arraylike(j) for j in key)
            if isinstance(key, tuple)
            else self._as_arraylike(key)
        )
        return self._return(self._data[i])

    def __setitem__(self, key: Any, value: Tensor | _ScalarLike) -> None:
        self._data[self._as_arraylike(key)] = self._as_arraylike(value)

    def __iter__(self) -> Tensor:
        self.__iterator = 0
        return self

    def __next__(self) -> Tensor | _ScalarLike:
        if self.__iterator < self.shape[0]:
            data = self[self.__iterator]
            self.__iterator += 1
            return data
        raise StopIteration

    def __add__(self, other: Tensor | _ScalarLike) -> Tensor:
        return self._return(self._data + self._as_arraylike(other))

    def __radd__(self, other: _ScalarLike) -> Tensor:
        return self + other

    def __mul__(self, other: Tensor | _ScalarLike) -> Tensor:
        return self._return(self._data * self._as_arraylike(other))

    def __rmul__(self, other: _ScalarLike) -> Tensor:
        return self * other

    def __pow__(self, other: Tensor | _ScalarLike) -> Tensor:
        return self._return(self._data ** self._as_arraylike(other))

    def __rpow__(self, other: _ScalarLike) -> Tensor:
        return tensor(other, self.device) ** self

    def __neg__(self) -> Tensor:
        return self * -1

    def __sub__(self, other: Tensor | _ScalarLike) -> Tensor:
        return self._return(self._data - self._as_arraylike(other))

    def __rsub__(self, other: _ScalarLike) -> Tensor:
        return -self + other

    def __truediv__(self, other: Tensor | _ScalarLike) -> Tensor:
        return self._return(self._data / self._as_arraylike(other))

    def __rtruediv__(self, other: _ScalarLike) -> Tensor:
        return self**-1 * other

    def __floordiv__(self, other: Tensor | _ScalarLike) -> Tensor:
        return self._return(self._data // self._as_arraylike(other))

    def __rfloordiv__(self, other: _ScalarLike) -> Tensor:
        return (other // self).astype(self.dtype)

    def __mod__(self, other: int) -> Tensor:
        return self._return(self._data % other)

    def __matmul__(self, other: Tensor) -> Tensor:
        return self._return(self._data @ other.data)

    def __lt__(self, other: Tensor | _ScalarLike) -> Tensor:
        return self._return(self._data < self._as_arraylike(other))

    def __gt__(self, other: Tensor | _ScalarLike) -> Tensor:
        return self._return(self._data > self._as_arraylike(other))

    def __le__(self, other: Tensor | _ScalarLike) -> Tensor:
        return self._return(self._data <= self._as_arraylike(other))

    def __ge__(self, other: Tensor | _ScalarLike) -> Tensor:
        return self._return(self._data >= self._as_arraylike(other))

    def __eq__(self, other: Tensor | _ScalarLike) -> Tensor:
        return self._return(self._data == self._as_arraylike(other))

    def __ne__(self, other: Tensor | _ScalarLike) -> Tensor:
        return self._return(self._data != self._as_arraylike(other))

    def __iadd__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data += self._as_arraylike(other)
        return self

    def __isub__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data -= self._as_arraylike(other)
        return self

    def __imul__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data *= self._as_arraylike(other)
        return self

    def __idiv__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data /= self._as_arraylike(other)
        return self

    def __ifloordiv__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data //= self._as_arraylike(other)
        return self

    def __imod__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data %= self._as_arraylike(other)
        return self

    def __ipow__(self, other: Tensor | _ScalarLike) -> Tensor:
        self._data **= self._as_arraylike(other)
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
        dtype : str
            Datatype of tensor elements.

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
    # STRUCTURE CHANGING METHODS
    # ----------------------------------------------------------------------------------------------

    def reshape(self, shape: _ShapeLike) -> Tensor:
        """Returns a view of the tensor tensor of a given shape.

        Parameters
        ----------
        ShapeLike
            Shape of the new tensor.

        Returns
        -------
        Tensor
            Reshaped tensor.
        """
        return Tensor(self._data.reshape(*shape))

    def flatten(self) -> Tensor:
        """Returns a flattened, one-dimensional tensor."""
        return Tensor(self._data.reshape((-1,)))

    def transpose(self, axes: tuple[int, int] = (-2, -1)) -> Tensor:
        """Transposes a tensor by swapping two axes.

        Parameters
        ----------
        axes : tuple[int, int], optional
            Transpose axes, by default (-2, -1).

        Returns
        -------
        Tensor
            Transposed tensor.
        """
        if self.ndim < 2:
            return self
        return self.moveaxis(from_axis=axes[0], to_axis=axes[1])

    def insert_dim(self, axis: _AxisLike) -> Tensor:
        """Returns a view of the tensor containing an added dimension at a given axis.

        Parameters
        ----------
        axis : AxisLike
            Where to insert the new dimension.

        Returns
        -------
        Tensor
            Tensor with an added dimension.
        """
        return Tensor(self._engine.expand_dims(self._data, axis=axis))

    def add_dims(self, target_dims: int) -> Tensor:
        """Returns a view of the tensor with added trailing dimensions.

        Parameters
        ----------
        target_dims : int
            Total number of dimensions needed.

        Returns
        -------
        Tensor
            Tensor with specified number of dimensions.
        """
        return self.reshape(self.shape + (1,) * (target_dims - self.ndim))

    def resize(self, shape: _ShapeLike) -> Tensor:
        """Returns a new tensor with the specified shape.
        If the new tensor is larger than the original one, it is filled with zeros.

        Parameters
        ----------
        shape : ShapeLike
            Shape of the new tensor.

        Returns
        -------
        Tensor
            Resized tensor.
        """
        return Tensor(self._engine.resize(self._data, shape))

    def repeat(self, n_repeats: int, axis: int) -> Tensor:
        """Repeat elements of a tensor.

        Parameters
        ----------
        n_repeats : int
            Number of repeats.
        axis : int
            Axis, along which the values are repeated.

        Returns
        -------
        Tensor
            Tensor with repeated values.
        """
        return Tensor(self._data.repeat(n_repeats, axis))

    def tile(self, n_repeats: int, axis: int) -> Tensor:
        """Repeat elements of a tensor.

        Parameters
        ----------
        n_repeats : int
            Number of repeats.
        axis : int
            Axis, along which the values are repeated.

        Returns
        -------
        Tensor
            Tensor with repeated values.
        """
        repeats = [1] * self.ndim
        repeats[axis] = n_repeats
        return Tensor(self._engine.tile(self._data, tuple(repeats)))

    def pad(self, padding: int | tuple[int, int] | tuple[tuple[int, int], ...]) -> Tensor:
        """Returns a padded tensor using zero padding.

        Parameters
        ----------
        pad_width : int | tuple[int, int] | tuple[tuple[int, int], ...]
            Padding width.
            `int`: Same padding for before and after in all dimensions.
            `tuple[int, int]`: Specific before and after padding in all dimensions.
            `tuple[tuple[int, int]`: Specific before and after padding for each dimension.

        Returns
        -------
        Tensor
            Padded tensor.
        """
        return Tensor(self._engine.pad(self._data, padding))

    def pad_to_shape(self, shape: _ShapeLike) -> Tensor:
        """Returns a padded tensor using zero padding that matches a specified shape.

        Parameters
        ----------
        shape : ShapeLike
            Final shape of the padded tensor.

        Returns
        -------
        Tensor
            Padded tensor.
        """
        padding = tuple((int(0), shape[i] - self.shape[i]) for i in range(self.ndim))
        return self.pad(padding)

    def moveaxis(self, from_axis: int, to_axis: int) -> Tensor:
        """Move axes of an array to new positions. Other axes remain in their original order.

        Parameters
        ----------
        from_axis : int
            Original positions of the axes to move. These must be unique.
        to_axis : int
            Destination positions for each of the original axes. These must also be unique.

        Returns
        -------
        Tensor
            Tensor with moved axes.
        """
        return Tensor(self._engine.moveaxis(self._data, from_axis, to_axis))

    def squeeze(self) -> Tensor:
        """Removes axis with length one from the tensor."""
        return Tensor(self._data.squeeze())

    def flip(self, axis: Optional[_AxisLike] = None) -> Tensor:
        """Returns a tensor with flipped elements along given axis.

        Parameters
        ----------
        axis : AxisLike, optional
            Axis alown which to flip the tensor, by default None.
            `None`: flip all the axes.
            `int`: flip given axis.
            `tuple[int, ...]`: flip all given axes.

        Returns
        -------
        Tensor
            Tensor containing flipped values.
        """
        return Tensor(self._engine.flip(self._data, axis=axis))

    # ----------------------------------------------------------------------------------------------
    # MEMORY/DEVICE METHODS
    # ----------------------------------------------------------------------------------------------

    def copy(self) -> Tensor:
        """Creates a copy of the tensor."""
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

    # ----------------------------------------------------------------------------------------------
    # OTHER OPERATIONS
    # ----------------------------------------------------------------------------------------------

    def sum(self, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
        """Sum of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike, optional
            Axis over which the sum is computed, by default None.
            If None it is computed over the flattened tensor.
        keepdims : bool, optional
            Whether to keep the tensors dimensions, by default False.
            If false the tensor is collapsed along the given axis.

        Returns
        -------
        Tensor
            Tensor containing the sum of elements.
        """
        return self._return(self._data.sum(axis=axis, keepdims=keepdims))

    def prod(self, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
        """Product of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike, optional
            Axis over which the product is computed, by default None.
            If None it is computed over the flattened tensor.
        keepdims : bool, optional
            Whether to keep the tensors dimensions, by default False.
            If false the tensor is collapsed along the given axis.

        Returns
        -------
        Tensor
            Tensor containing the product of elements.
        """
        return self._return(self._data.prod(axis=axis, keepdims=keepdims))

    def mean(self, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
        """Mean of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike, optional
            Axis over which the mean is computed, by default None.
            If None it is computed over the flattened tensor.
        keepdims : bool, optional
            Whether to keep the tensors dimensions, by default False.
            If false the tensor is collapsed along the given axis.

        Returns
        -------
        Tensor
            Tensor containing the mean of elements.
        """
        return self._return(self._data.mean(axis=axis, keepdims=keepdims))

    def var(
        self, axis: Optional[_AxisLike] = None, ddof: int = 0, keepdims: bool = False
    ) -> Tensor | _ScalarLike:
        """Variance of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike, optional
            Axis over which the variance is computed, by default None.
            If None it is computed over the flattened tensor.
        ddof : int, optional
            "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof,
            where N represents the number of elements, by default 0.
        keepdims : bool, optional
            Whether to keep the tensors dimensions, by default False.
            If false the tensor is collapsed along the given axis.

        Returns
        -------
        Tensor
            Tensor containing the variance of elements.
        """
        return self._return(self._data.var(axis=axis, ddof=ddof, keepdims=keepdims))

    def std(self, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
        """Standard deviation of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike, optional
            Axis over which the standard deviation is computed, by default None.
            If None it is computed over the flattened tensor.
        keepdims : bool, optional
            Whether to keep the tensors dimensions, by default False.
            If false the tensor is collapsed along the given axis.

        Returns
        -------
        Tensor
            Tensor containing the standard deviation of elements.
        """
        return self._return(self._data.std(axis=axis, keepdims=keepdims))

    def min(self, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
        """Minimum of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike, optional
            Axis over which the minimum is computed, by default None.
            If None it is computed over the flattened tensor.
        keepdims : bool, optional
            Whether to keep the tensors dimensions, by default False.
            If false the tensor is collapsed along the given axis.

        Returns
        -------
        Tensor
            Tensor containing the minimum of elements.
        """
        return self._return(self._data.min(axis=axis, keepdims=keepdims))

    def max(self, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
        """Maximum of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike, optional
            Axis over which the maximum is computed, by default None.
            If none it is computed over the flattened tensor.
        keepdims : bool, optional
            Whether to keep the tensors dimensions, by default False.
            If false the tensor is collapsed along the given axis.

        Returns
        -------
        Tensor
            Tensor containing the maximum of elements.
        """
        return self._return(self._data.max(axis=axis, keepdims=keepdims))

    def argmax(self, axis: Optional[_AxisLike] = None, keepdims: bool = False) -> Tensor:
        """Returns the indices of maximum values along a given axis.

        Parameters
        ----------
        axis : Optional[AxisLike] = None
            Axes, along which the maximum value is located, by default None.
            If None it is computed over the flattened tensor.
        keepdims : bool, optional
            Whether to keep the tensors dimensions, by default False.
            If false the tensor is collapsed along the given axis.

        Returns
        -------
        Tensor
            Tensor containing indices.
        """
        return self._return(self._engine.argmax(self._data, axis=axis, keepdims=keepdims))

    def round(self, decimals: int) -> Tensor:
        """Rounds the value of tensor elements.

        Parameters
        ----------
        decimals : int
            Decimal places of rounded values.

        Returns
        -------
        Tensor
            Tensor containing the rounded values.
        """
        return Tensor(self._data.round(decimals))

    def exp(self) -> Tensor:
        """Exponential of tensor element."""
        return Tensor(self._engine.exp(self._data))

    def log(self) -> Tensor:
        """Natural logarithm of tensor elements."""
        return Tensor(self._engine.log(self._data))

    def log10(self) -> Tensor:
        """Logarithm with base 10 of tensor elements."""
        return Tensor(self._engine.log10(self._data))

    def log2(self) -> Tensor:
        """Logarithm with base 2 of tensor elements."""
        return Tensor(self._engine.log2(self._data))

    def sin(self) -> Tensor:
        """Sine of tensor elements."""
        return Tensor(self._engine.sin(self._data))

    def sinh(self) -> Tensor:
        """Hyperbolic sine of tensor elements."""
        return Tensor(self._engine.sinh(self._data))

    def cos(self) -> Tensor:
        """Cosine of tensor elements."""
        return Tensor(self._engine.cos(self._data))

    def cosh(self) -> Tensor:
        """Hyperbolic cosine of tensor elements."""
        return Tensor(self._engine.cosh(self._data))

    def tan(self) -> Tensor:
        """Tangent of tensor elements."""
        return Tensor(self._engine.tan(self._data))

    def tanh(self) -> Tensor:
        """Hyperbolical tangent of tensor elements."""
        return Tensor(self._engine.tanh(self._data))

    def sech(self) -> Tensor:
        """Hyperbolic secant of tensor elements."""
        return self.cosh() ** -1

    def abs(self) -> Tensor:
        """Absolute values of tensor elements."""
        return Tensor(self._engine.abs(self._data))

    def sqrt(self) -> Tensor:
        """Square root of tensor elements."""
        return Tensor(self._engine.sqrt(self._data))

    def fft1d(
        self,
        n: Optional[int] = None,
        axis: int = -1,
        dtype: Optional[_ComplexLike] = None,
    ) -> Tensor:
        """Computes the 1D Fast Fourier Transform over a specified axis.

        Parameters
        ----------
        n : int, optional
            Length of the transformed axis of the output, by default None.
        axis : int, optional
            Axis over which to compute the FFT, by default -1.
        dtype : ComplexLike, optional
            Datatype of the output tensor, by default None.

        Returns
        -------
        Tensor
            Complex tensor.
        """
        return tensor(
            self._engine.fft.fft(self._data, n=n, axis=axis), device=self.device, dtype=dtype
        )

    def ifft1d(
        self,
        n: Optional[int] = None,
        axis: int = -1,
        dtype: Optional[_ComplexLike] = None,
    ) -> Tensor:
        """Computes the inverse 1D Fast Fourier Transform over a specified axis.

        Parameters
        ----------
        n : int, optional
            Length of the transformed axis of the output, by default None.
        axis : int, optional
            Axis over which to compute the inverse FFT, by default -1.
        dtype : FloatLike, optional
            Datatype of the output tensor, by default None.

        Returns
        -------
        Tensor
            Float tensor.
        """
        return tensor(
            self._engine.fft.ifft(self._data, n=n, axis=axis), device=self.device, dtype=dtype
        )

    def fft2d(
        self,
        s: Optional[_ShapeLike] = None,
        axes: tuple[int, int] = (-2, -1),
        dtype: Optional[_ComplexLike] = None,
    ) -> Tensor:
        """Computes the 2D Fast Fourier Transform over two specified axes.

        Parameters
        ----------
        n : ShapeLike, optional
            Shape (length of each transformed axis) of the output, by default None.
        axes : tuple[int, int], optional
            Axes over which to compute the FFT, by default (-2, -1).
        dtype : ComplexLike, optional
            Datatype of the output tensor, by default None.

        Returns
        -------
        Tensor
            Complex tensor.
        """
        return tensor(
            self._engine.fft.fft2(self._data, s=s, axes=axes), device=self.device, dtype=dtype
        )

    def ifft2d(
        self,
        s: Optional[_ShapeLike] = None,
        axes: tuple[int, int] = (-2, -1),
        dtype: Optional[_ComplexLike] = None,
    ) -> Tensor:
        """Applies the inverse 1D Fast Fourier Transform to the tensor.

        Parameters
        ----------
        n : ShapeLike, optional
            Shape (length of each transformed axis) of the output, by default None.
        axes : tuple[int, int], optional
            Axes over which to compute the inverse FFT, by default (-2, -1).
        dtype : ComplexLike, optional
            Datatype of the output tensor, by default None.

        Returns
        -------
        Tensor
            Complex tensor.
        """
        return tensor(
            self._engine.fft.ifft2(self._data, s=s, axes=axes), device=self.device, dtype=dtype
        )

    def real(self, dtype: Optional[_DtypeLike] = None) -> Tensor:
        """Returns the real parts of the complex tensor.

        Parameters
        ----------
        dtype : DtypeLike, optional
            Datatype of the output tensor, by default None.

        Returns
        -------
        Tensor
            Tensor containing real values.
        """
        return tensor(self._engine.real(self._data), device=self.device, dtype=dtype)

    def append(self, values: Tensor, axis: int = -1) -> Tensor:
        """Returns a copy of the tensor with appended values.

        Parameters
        ----------
        values : Tensor
            Values to append.
        axis : int, optional
            Axis alowng which to append the values, by default -1.

        Returns
        -------
        Tensor
            Tensor containing appended values.
        """
        return Tensor(self._engine.append(self._data, values.data, axis=axis))

    def clip(
        self, min_value: Optional[_ScalarLike] = None, max_value: Optional[_ScalarLike] = None
    ) -> Tensor:
        """Limits the values of a tensor.

        Parameters
        ----------
        min_value : ScalarLike, optional
            Lower bound, by default None. If None, no clipping is performed on this edge.
        max_value : ScalarLike
            Upper bound, by default None. If None, no clipping is performed on this edge.

        Returns
        -------
        Tensor
            Tensor containing clipped values.
        """
        return Tensor(self._engine.clip(self._data, min_value, max_value))

    def unique(self) -> Tensor:
        """Returns the unique ordered values of the tensor."""
        return Tensor(self._engine.unique(self._data))

    def split(self, splits: int | Sequence[int], axis: int = -1) -> list[Tensor]:
        """Returns a list of new tensors by splitting the tensor.

        Parameters
        ----------
        splits : int | list[int]
            `int`: tensor is split into n equally sized tensors.
            `Sequence[int]`: tensor is split at the given indices.
        axis : int, optional
            Axis along which to split the tensor, by default -1.

        Returns
        -------
        list[Tensor]
            List of tensors containing the split data.
        """
        return [Tensor(s) for s in self._engine.split(self._data, splits, axis=axis)]

    def get_diagonal(self, d: int = 0) -> Tensor:
        """Extract a diagonal or construct a diagonal tensor.

        Parameters
        ----------
        d : int, optional
            Index of the diagonal, by default 0.
            `0`: main diagonal
            `> 0`: above main diagonal
            `< 0`: below main diagonal

        Returns
        -------
        Tensor
            The extracted diagonal or constructed diagonal tensor.
        """
        return Tensor(self._engine.diag(self._data, k=d))

    def tril(self, d: int = 0) -> Tensor:
        """Returns the lower triangle of a tensor below the
        d-th diagonal of the last two dimensions.

        Parameters
        ----------
        d : int, optional
            Index of the diagonal, by default 0.
            `0`: main diagonal
            `> 0`: above main diagonal
            `< 0`: below main diagonal

        Returns
        -------
        Tensor
            Lower triangle tensor.
        """
        return Tensor(self._engine.tril(self._data, k=d))

    def triu(self, d: int = 0) -> Tensor:
        """Returns the upper triangle of a tensor above the
        d-th diagonal of the last two dimensions.

        Parameters
        ----------
        d : int, optional
            Index of the diagonal, by default 0.
            `0`: main diagonal
            `> 0`: above main diagonal
            `< 0`: below main diagonal

        Returns
        -------
        Tensor
            Upper triangle tensor.
        """
        return Tensor(self._engine.triu(self._data, k=d))

    def _as_arraylike(self, value: Any) -> _ArrayLike:
        if isinstance(value, Tensor):
            return value.data
        return value

    def _return(self, value: _ArrayLike | _ScalarLike) -> Tensor:
        if isinstance(value, _ArrayLike):
            return Tensor(value)
        return tensor(value, device=self.device)
