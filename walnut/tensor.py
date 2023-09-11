"""Tensor module"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import cupy as cp


__all__ = ["Tensor", "ShapeError"]


ShapeLike = tuple[int, ...]
AxisLike = int | tuple[int, ...]
ArrayLike = np.ndarray | cp.ndarray
NpTypeLike = np.float32 | np.float64 | np.int32 | np.int64
PyTypeLike = list | float | int


class ShapeError(Exception):
    """Incompatible tensor shapes."""


@dataclass(init=False, repr=False, slots=True)
class Tensor:
    """Tensor object."""

    _data: ArrayLike
    _grad: ArrayLike | None
    _iterator: int
    _device: str
    temp_params: dict[str, ArrayLike]

    def __init__(
        self,
        data: ArrayLike | NpTypeLike | PyTypeLike,
        dtype: str = "float32",
        copy: bool = False,
        device: str = "cpu",
    ) -> None:
        """Tensor object.

        Parameters
        ----------
        data : NpArrayLike | NpTypeLike | PyTypeLike
            Data to initialize the tensor.
        dtype: str, optional
            Datatype of the tensor data, by default "float32".
        copy: bool, optional
            If true, the data object is copied (may impact performance), by default False.
        device: str, optinal
            The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".
        """

        if device == "cuda":
            # create cupy array on GPU
            self.data = cp.array(data, copy=copy, dtype=dtype)
        else:
            # create numpy array on CPU
            self.data = np.array(data, copy=copy, dtype=dtype)

        self._grad = None
        self.temp_params = {}
        self._iterator = 0
        self._device = device

    # ----------------------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------------------

    @property
    def data(self) -> ArrayLike:
        """Tensor data."""
        return self._data

    @data.setter
    def data(self, value: ArrayLike) -> None:
        if not isinstance(value, ArrayLike):
            raise ValueError("Invalid dtype.")
        self._data = value

    @property
    def grad(self) -> ArrayLike | None:
        """Tensor gradient."""
        return self._grad

    @grad.setter
    def grad(self, value: ArrayLike | None) -> None:
        if value is not None and not isinstance(value, ArrayLike):
            raise ValueError("Invalid dtype.")
        if value is not None and value.shape != self.shape:
            raise ShapeError(f"Grad shape {value.shape} != data shape {self.shape}")
        if value is None:
            self._grad = value
        elif self.device == "cpu":
            self._grad = np.array(value, copy=False, dtype="float32")
        else:
            self._grad = cp.array(value, copy=False, dtype="float32")

    @property
    def shape(self) -> ShapeLike:
        """Tensor shape."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Tensor dimensions."""
        return self.data.ndim

    @property
    def len(self) -> int:
        """Tensor length along axis 0."""
        return len(self.data) if self.data.ndim > 0 else 0

    @property
    def T(self) -> ArrayLike:
        """Tensor data transposed."""
        return self.data.T

    @property
    def dtype(self) -> str:
        """Tensor data datatype."""
        return str(self.data.dtype)

    @property
    def device(self) -> str:
        """Storage device."""
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        if value not in ("cpu", "cuda"):
            raise AttributeError("Unknown device.")
        self._device = value

    def to_device(self, device: str) -> None:
        """Moves the tensor to a specified device.

        Parameters
        ----------
        device : str
            Device to move the tensor to. Valid options are "cpu" and "cuda".

        Raises
        ----------
        AttributeError
            If device is not "cpu" or "cuda".

        """
        if device not in ("cpu", "cuda"):
            raise AttributeError("Unknown device.")
        if self._device == device:
            return

        self._device = device
        if device == "cpu":
            self.data = cp.asnumpy(self.data)
            if self.grad is not None:
                self.grad = cp.asnumpy(self.grad)
        else:
            self.data = cp.array(self.data)
            if self.grad is not None:
                self.grad = cp.array(self.grad)

    # ----------------------------------------------------------------------------------------------
    # OVERLOADS
    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            self.data.__repr__()
            .replace("array", "tnsor")
            .replace(", dtype=float32", "")
        )

    def __call__(self) -> ArrayLike:
        return self.data

    def __getitem__(self, key) -> Tensor:
        if isinstance(key, Tensor):
            return Tensor(self.data[key.data], dtype=self.dtype, device=self.device)
        return Tensor(self.data[key], dtype=self.dtype, device=self.device)

    def __setitem__(self, key, value) -> None:
        if isinstance(key, Tensor) and isinstance(value, Tensor):
            self.data[key.data] = value.data
        elif isinstance(key, Tensor):
            self.data[key.data] = value
        elif isinstance(value, Tensor):
            self.data[key] = value.data
        else:
            self.data[key] = value

    def __iter__(self):
        self._iterator = 0
        return self

    def __next__(self):
        if self._iterator < self.len:
            ret = Tensor(
                self.data[self._iterator], dtype=self.dtype, device=self.device
            )
            self._iterator += 1
            return ret
        raise StopIteration

    def __add__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data + other.data, device=self.device)

    def __mul__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data * other.data, device=self.device)

    def __sub__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data - other.data, device=self.device)

    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data / other.data, device=self.device)

    def __floordiv__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data // other.data, device=self.device)

    def __pow__(self, other: int | float) -> Tensor:
        return Tensor(self.data**other, device=self.device)

    def __mod__(self, other: int) -> Tensor:
        return Tensor(self.data % other, device=self.device)

    def __matmul__(self, other: Tensor) -> Tensor:
        return Tensor(self.data @ other.data, device=self.device)

    def __lt__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data < other.data, dtype="int", device=self.device)

    def __gt__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data > other.data, dtype="int", device=self.device)

    def __le__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data <= other.data, dtype="int", device=self.device)

    def __ge__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data >= other.data, dtype="int", device=self.device)

    def __eq__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data == other.data, dtype="int", device=self.device)

    def __ne__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data != other.data, dtype="int", device=self.device)

    def __isub__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data += other.data
        return Tensor(self.data, device=self.device)

    def __iadd__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data += other.data
        return Tensor(self.data, device=self.device)

    def __imul__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data *= other.data
        return Tensor(self.data, device=self.device)

    def __idiv__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data /= other.data
        return Tensor(self.data, device=self.device)

    def __ifloordiv__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data //= other.data
        return Tensor(self.data, device=self.device)

    def __imod__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data %= other.data
        return Tensor(self.data, device=self.device)

    def __ipow__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data **= other.data
        return Tensor(self.data, device=self.device)

    def __neg__(self) -> Tensor:
        self.data = -1.0 * self.data
        return Tensor(self.data, device=self.device)

    def __tensorify(self, other: Tensor | float | int) -> Tensor:
        if not isinstance(other, Tensor):
            return Tensor(other, device=self.device)
        return other

    def __len__(self) -> int:
        return len(self.data)

    # ----------------------------------------------------------------------------------------------
    # FUNCTIONS
    # ----------------------------------------------------------------------------------------------

    def sum(self, axis: AxisLike | None = None, keepdims: bool = False) -> Tensor:
        """Sum of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike | None, optional
            Axis over which the sum is computed, by default None.
            If none it is computed over the flattened tensor.
        keepdims : bool, optional
            Whether to keep the tensors dimensions, by default False.
            If false the tensor is collapsed along the given axis.

        Returns
        -------
        Tensor
            Tensor containing the sum of elements.
        """
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), device=self.device)

    def mean(self, axis: AxisLike | None = None, keepdims: bool = False) -> Tensor:
        """Mean of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike | None, optional
            Axis over which the mean is computed, by default None.
            If none it is computed over the flattened tensor.
        keepdims : bool, optional
            Whether to keep the tensors dimensions, by default False.
            If false the tensor is collapsed along the given axis.

        Returns
        -------
        Tensor
            Tensor containing the mean of elements.
        """
        return Tensor(self.data.mean(axis=axis, keepdims=keepdims), device=self.device)

    def var(
        self, axis: AxisLike | None = None, ddof: int = 0, keepdims: bool = False
    ) -> Tensor:
        """Variance of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike | None, optional
            Axis over which the variance is computed, by default None.
            If none it is computed over the flattened tensor.
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
        return Tensor(
            self.data.var(axis=axis, ddof=ddof, keepdims=keepdims),
            device=self.device,
        )

    def std(self, axis: AxisLike | None = None, keepdims: bool = False) -> Tensor:
        """Standard deviation of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike | None, optional
            Axis over which the standard deviation is computed, by default None.
            If none it is computed over the flattened tensor.
        keepdims : bool, optional
            Whether to keep the tensors dimensions, by default False.
            If false the tensor is collapsed along the given axis.

        Returns
        -------
        Tensor
            Tensor containing the standard deviation of elements.
        """
        return Tensor(self.data.std(axis=axis, keepdims=keepdims), device=self.device)

    def min(self, axis: AxisLike | None = None, keepdims: bool = False) -> Tensor:
        """Minimum of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike | None, optional
            Axis over which the minimum is computed, by default None.
            If none it is computed over the flattened tensor.
        keepdims : bool, optional
            Whether to keep the tensors dimensions, by default False.
            If false the tensor is collapsed along the given axis.

        Returns
        -------
        Tensor
            Tensor containing the minimum of elements.
        """
        return Tensor(
            self.data.min(axis=axis, keepdims=keepdims),
            dtype=self.dtype,
            device=self.device,
        )

    def max(self, axis: AxisLike | None = None, keepdims: bool = False) -> Tensor:
        """Maximum of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike | None, optional
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
        return Tensor(
            self.data.max(axis=axis, keepdims=keepdims),
            dtype=self.dtype,
            device=self.device,
        )

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
        return Tensor(self.data.round(decimals), device=self.device)

    def exp(self) -> Tensor:
        """Exponential of tensor elements.

        Returns
        -------
        Tensor
            Tensor containing the value of e**x for each element.
        """
        tmp = np.exp(self.data) if self.device == "cpu" else cp.exp(self.data)
        return Tensor(tmp, device=self.device)

    def log(self) -> Tensor:
        """Natural logarithm of tensor elements.

        Returns
        -------
            Tensor
                Tensor containing the value of log(x) for each element.
        """
        tmp = np.log(self.data) if self.device == "cpu" else cp.log(self.data)
        return Tensor(tmp, device=self.device)

    def log10(self) -> Tensor:
        """Logarithm with base 10 of tensor elements.

        Returns
        -------
            Tensor
                Tensor containing the value of log10(x) for each element.
        """
        tmp = np.log10(self.data) if self.device == "cpu" else cp.log10(self.data)
        return Tensor(tmp, device=self.device)

    def log2(self) -> Tensor:
        """Logarithm with base 2 of tensor elements.

        Returns
        -------
            Tensor
                Tensor containing the value of log2(x) for each element.
        """
        tmp = np.log2(self.data) if self.device == "cpu" else cp.log2(self.data)
        return Tensor(tmp, device=self.device)

    def tanh(self) -> Tensor:
        """Hyperbolical tangent of tensor elements.

        Returns
        -------
            Tensor
            Tensor containing the value of tanh(x) for each element.
        """
        tmp = np.tanh(self.data) if self.device == "cpu" else cp.tanh(self.data)
        return Tensor(tmp, device=self.device)

    def abs(self) -> Tensor:
        """Absolute values of tensor elements.

        Returns
        -------
            Tensor
            Tensor containing the absolute value for each element.
        """
        tmp = np.abs(self.data) if self.device == "cpu" else cp.abs(self.data)
        return Tensor(tmp, dtype=self.dtype, device=self.device)

    def sqrt(self) -> Tensor:
        """Square root of tensor elements.

        Returns
        -------
            Tensor
            Tensor containing the square root value for each element.
        """
        tmp = np.sqrt(self.data) if self.device == "cpu" else cp.sqrt(self.data)
        return Tensor(tmp, dtype=self.dtype, device=self.device)

    def item(self) -> float:
        """Returns the scalar value of the tensor data.

        Returns
        -------
        float
            Scalar of the tensor data.
        """
        return self.data.item()

    def reshape(self, shape: ShapeLike) -> Tensor:
        """Returns a reshaped tensor of data to fit a given shape.

        Parameters
        ----------
        ShapeLike
            Shape of the new tensor.

        Returns
        -------
        Tensor
            Reshapded tensor.
        """
        return Tensor(self.data.reshape(*shape), dtype=self.dtype, device=self.device)

    def pad(self, widths: tuple[tuple[int, ...]]) -> Tensor:
        """Returns a padded tensor using zero padding.

        Parameters
        ----------
        widths : tuple[tuple[int, ...]]
            Padding widths for each dimension of the tensor.

        Returns
        -------
        Tensor
            Padded tensor.
        """
        if self.device == "cpu":
            tmp = np.pad(self.data, widths)
        else:
            tmp = cp.pad(self.data, widths)
        return Tensor(tmp, dtype=self.dtype, device=self.device)

    def flatten(self) -> Tensor:
        """Returns a flattened, one-dimensional tensor.

        Returns
        -------
        Tensor
            Flattened, one-dimensional version of the tensor.
        """
        return Tensor(self.data.reshape((-1,)), dtype=self.dtype, device=self.device)

    def transpose(self, axis: AxisLike | None = None) -> Tensor:
        """Transposes a tensor along given axes.

        Parameters
        ----------
        axes : AxisLike, optional
            Permutation of axes of the transposed tensor, by default None.

        Returns
        -------
        Tensor
            Transposed tensor.
        """
        if self.device == "cpu":
            tmp = np.transpose(self.data, axes=axis)
        else:
            tmp = cp.transpose(self.data, axes=axis)
        return Tensor(tmp, dtype=self.dtype, device=self.device)

    def astype(self, dtype: str) -> Tensor:
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
        return Tensor(self.data, dtype=dtype, device=self.device)

    def append(self, values: Tensor, axis: int) -> Tensor:
        """Returns a copy of the tensor with appended values.

        Parameters
        ----------
        values : Tensor
            Values to append.
        axis : int
            Axis alown which to append the values

        Returns
        -------
        Tensor
            Tensor containing appended values.
        """
        if self.device == "cpu":
            tmp = np.append(self.data, values.data, axis=axis)
        else:
            tmp = cp.append(self.data, values.data, axis=axis)
        return Tensor(tmp, dtype=self.dtype, device=self.device)

    def flip(self, axis: AxisLike) -> Tensor:
        """Returns a tensor with flipped elements along given axis.

        Parameters
        ----------
        axis : AxisLike
            Axis alown which to flip the tensor.

        Returns
        -------
        Tensor
            Tensor containing flipped values.
        """
        if self.device == "cpu":
            tmp = np.flip(self.data, axis=axis)
        else:
            tmp = cp.flip(self.data, axis=axis)
        return Tensor(tmp, dtype=self.dtype, device=self.device)

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
        if self.device == "cpu":
            tmp = np.moveaxis(self.data, from_axis, to_axis)
        else:
            tmp = cp.moveaxis(self.data, from_axis, to_axis)
        return Tensor(tmp, dtype=self.dtype, device=self.device)

    def squeeze(self) -> Tensor:
        """Removes axis with length one from the tensor."""
        return Tensor(self.data.squeeze(), dtype=self.dtype, device=self.device)

    def argmax(self, axis: int | None = None) -> Tensor:
        """Returns the indices of maximum values along a given axis.

        Parameters
        ----------
        axis : int | None = None
            Axis, along which the maximum value is located, by default None.

        Returns
        -------
        int | tuple [int, ...]
            Index tensor.
        """
        if self.device == "cpu":
            tmp = np.argmax(self.data, axis=axis)
        else:
            tmp = cp.argmax(self.data, axis=axis)
        return Tensor(tmp, dtype="int", device=self.device)

    def resize(self, shape: ShapeLike) -> Tensor:
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
        if self.device == "cpu":
            tmp = np.resize(self.data, shape)
        else:
            tmp = cp.resize(self.data, shape)
        return Tensor(tmp, dtype=self.dtype, device=self.device)

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
        return Tensor(
            self.data.repeat(n_repeats, axis), dtype=self.dtype, device=self.device
        )

    def clip(self, min_value: int | float, max_value: int | float) -> Tensor:
        """Limits the values of a tensor.

        Parameters
        ----------
        min_value : int | float
            Lower bound of allowed values.
        max_value : int | float
            Upper bound of allowed values.

        Returns
        -------
        Tensor
            Tensor containing clipped values.
        """
        if self.device == "cpu":
            tmp = np.clip(self.data, min_value, max_value)
        else:
            tmp = cp.clip(self.data, min_value, max_value)
        return Tensor(tmp, dtype=self.dtype, device=self.device)
