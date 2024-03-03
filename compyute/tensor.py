"""Tensor module"""

from __future__ import annotations

from compyute.engine import (
    get_engine,
    numpy_to_cupy,
    cupy_to_numpy,
    ArrayLike,
    ScalarLike,
)


__all__ = ["Tensor", "tensor", "ShapeError"]
ShapeLike = tuple[int, ...]
AxisLike = int | tuple[int, ...]


class ShapeError(Exception):
    """Incompatible tensor shapes."""


def tensor(
    data: ArrayLike | ScalarLike, copy: bool = False, device: str = "cpu"
) -> Tensor:
    """Creates a Tensor instance and infers the dtype automatically.

    Parameters
    ----------
    data : ArrayLike | ScalarLike
        Data to initialize the tensor.
    copy : bool, optional
        If true, the data object is copied (may impact performance), by default False.
    device : str, optional
        The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".

    Returns
    -------
    Tensor
        Tensor object.
    """
    if not isinstance(data, ArrayLike):
        data = get_engine(device).array(data)
    return Tensor(data, dtype=data.dtype, copy=copy, device=device)


class Tensor:
    """Tensor object."""

    def __init__(
        self,
        data: ArrayLike | ScalarLike,
        dtype: str = "float64",
        copy: bool = False,
        device: str = "cpu",
    ) -> None:
        """Tensor object.

        Parameters
        ----------
        data : ArrayLike | ScalarLike
            Data to initialize the tensor.
        dtype: str, optional
            Datatype of the tensor data, by default "float64".
        copy: bool, optional
            If true, the data object is copied (may impact performance), by default False.
        device: str, optinal
            The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".
        """

        self._data = get_engine(device).array(data, copy=copy, dtype=dtype)
        self._grad = None
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
            raise ValueError("Can only set the gradient to be an array.")
        self._data = value

    @property
    def grad(self) -> ArrayLike | None:
        """Tensor gradient."""
        return self._grad

    @grad.setter
    def grad(self, value: ArrayLike | None) -> None:
        if value is not None and not isinstance(value, ArrayLike):
            raise ValueError("Can only set the gradient to be an array or None.")
        if value is None:
            self._grad = None
        else:
            engine = get_engine(self.device)
            self._grad = engine.array(value, copy=False, dtype=value.dtype)

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
            self.data = cupy_to_numpy(self.data)
            if self.grad is not None:
                self.grad = cupy_to_numpy(self.grad)
        else:
            self.data = numpy_to_cupy(self.data)
            if self.grad is not None:
                self.grad = numpy_to_cupy(self.grad)

    def cpu(self):
        """Returns a copy of the tensor on the cpu."""
        if self.device == "cpu":
            return self
        t = Tensor(cupy_to_numpy(self.data), self.dtype, device="cpu")
        t.grad = cupy_to_numpy(self.grad) if self.grad is not None else None
        return t

    def cuda(self):
        """Returns a copy of the tensor on the gpu."""
        if self.device == "cuda":
            return self
        t = Tensor(numpy_to_cupy(self.data), self.dtype, device="cuda")
        t.grad = numpy_to_cupy(self.grad) if self.grad is not None else None
        return t

    # ----------------------------------------------------------------------------------------------
    # OVERLOADS
    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        return self.data.__repr__().replace("array", "tnsor")

    def __call__(self) -> ArrayLike:
        return self.data

    def __getitem__(self, key) -> Tensor:
        if isinstance(key, tuple):
            new_key = tuple(e.data if isinstance(e, Tensor) else e for e in key)
        elif isinstance(key, Tensor):
            new_key = key.data
        else:
            new_key = key
        return Tensor(self.data[new_key], dtype=self.dtype, device=self.device)

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
        if self._iterator < len(self):
            data = self.data[self._iterator]
            if isinstance(data, ArrayLike):
                data = Tensor(data, dtype=self.dtype, device=self.device)
            self._iterator += 1
            return data
        raise StopIteration

    # basic arithmetic
    def __add__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        r = self.data + other.data
        return Tensor(r, dtype=r.dtype, device=self.device)

    def __radd__(self, other: Tensor | float | int) -> Tensor:
        return self + other

    def __mul__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        r = self.data * other.data
        return Tensor(r, dtype=r.dtype, device=self.device)

    def __rmul__(self, other: Tensor | float | int) -> Tensor:
        return self * other

    def __pow__(self, other: int | float) -> Tensor:
        if "int" in self.dtype and other < 0:
            r = self.data.astype("float64") ** other
        else:
            r = self.data**other
        return Tensor(r, dtype=r.dtype, device=self.device)

    def __neg__(self) -> Tensor:
        return self * -1

    # derived arithmetic
    def __sub__(self, other: Tensor | float | int) -> Tensor:
        return self + (-other)

    def __rsub__(self, other: Tensor | float | int) -> Tensor:
        return other + (-self)

    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        return self * other**-1

    def __rtruediv__(self, other: Tensor | float | int) -> Tensor:
        return other * self**-1

    def __floordiv__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        r = self.data // other.data
        return Tensor(r, dtype=r.dtype, device=self.device)

    def __rfloordiv__(self, other: Tensor | float | int) -> Tensor:
        return other // self

    def __mod__(self, other: int) -> Tensor:
        r = self.data % other
        return Tensor(r, dtype=r.dtype, device=self.device)

    def __matmul__(self, other: Tensor) -> Tensor:
        r = self.data @ other.data
        return Tensor(r, dtype=r.dtype, device=self.device)

    # boolean operations
    def __lt__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        r = self.data < other.data
        return Tensor(r, dtype=r.dtype, device=self.device)

    def __gt__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        r = self.data > other.data
        return Tensor(r, dtype=r.dtype, device=self.device)

    def __le__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        r = self.data <= other.data
        return Tensor(r, dtype=r.dtype, device=self.device)

    def __ge__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        r = self.data >= other.data
        return Tensor(r, dtype=r.dtype, device=self.device)

    def __eq__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        r = self.data == other.data
        return Tensor(r, dtype=r.dtype, device=self.device)

    def __ne__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        r = self.data != other.data
        return Tensor(r, dtype=r.dtype, device=self.device)

    # inplace operations
    def __isub__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data += other.data
        return self

    def __iadd__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data += other.data
        return self

    def __imul__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data *= other.data
        return self

    def __idiv__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data /= other.data
        return self

    def __ifloordiv__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data //= other.data
        return self

    def __imod__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data %= other.data
        return self

    def __ipow__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data **= other.data
        return self

    def __tensorify(self, other: Tensor | float | int) -> Tensor:
        if not isinstance(other, Tensor):
            return Tensor(other, dtype=self.dtype, device=self.device)
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
        r = self.data.sum(axis=axis, keepdims=keepdims)
        return Tensor(r, dtype=r.dtype, device=self.device)

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
        r = self.data.mean(axis=axis, keepdims=keepdims)
        return Tensor(r, dtype=r.dtype, device=self.device)

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
        r = self.data.var(axis=axis, ddof=ddof, keepdims=keepdims)
        return Tensor(r, dtype=r.dtype, device=self.device)

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
        r = self.data.std(axis=axis, keepdims=keepdims)
        return Tensor(r, dtype=r.dtype, device=self.device)

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
        r = self.data.min(axis=axis, keepdims=keepdims)
        return Tensor(r, dtype=r.dtype, device=self.device)

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
        r = self.data.max(axis=axis, keepdims=keepdims)
        return Tensor(r, dtype=r.dtype, device=self.device)

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
        r = self.data.round(decimals)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def exp(self) -> Tensor:
        """Exponential of tensor elements.

        Returns
        -------
        Tensor
            Tensor containing the value of e**x for each element.
        """
        r = get_engine(self.device).exp(self.data)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def log(self) -> Tensor:
        """Natural logarithm of tensor elements.

        Returns
        -------
            Tensor
                Tensor containing the value of log(x) for each element.
        """
        r = get_engine(self.device).log(self.data)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def log10(self) -> Tensor:
        """Logarithm with base 10 of tensor elements.

        Returns
        -------
            Tensor
                Tensor containing the value of log10(x) for each element.
        """
        r = get_engine(self.device).log10(self.data)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def log2(self) -> Tensor:
        """Logarithm with base 2 of tensor elements.

        Returns
        -------
            Tensor
                Tensor containing the value of log2(x) for each element.
        """
        r = get_engine(self.device).log2(self.data)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def sin(self) -> Tensor:
        """Sine of tensor elements.

        Returns
        -------
        Tensor
            Tensor containing the value of sin(x) for each element.
        """
        r = get_engine(self.device).sin(self.data)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def sinh(self) -> Tensor:
        """Hyperbolic sine of tensor elements.

        Returns
        -------
        Tensor
            Tensor containing the value of sinh(x) for each element.
        """
        r = get_engine(self.device).sinh(self.data)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def cos(self) -> Tensor:
        """Cosine of tensor elements.

        Returns
        -------
        Tensor
            Tensor containing the value of cos(x) for each element.
        """
        r = get_engine(self.device).cos(self.data)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def cosh(self) -> Tensor:
        """Hyperbolic cosine of tensor elements.

        Returns
        -------
        Tensor
            Tensor containing the value of cosh(x) for each element.
        """
        r = get_engine(self.device).cosh(self.data)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def tan(self) -> Tensor:
        """Tangent of tensor elements.

        Returns
        -------
        Tensor
            Tensor containing the value of tan(x) for each element.
        """
        r = get_engine(self.device).tan(self.data)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def tanh(self) -> Tensor:
        """Hyperbolical tangent of tensor elements.

        Returns
        -------
            Tensor
            Tensor containing the value of tanh(x) for each element.
        """
        r = get_engine(self.device).tanh(self.data)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def sech(self) -> Tensor:
        """Hyperbolic secant of tensor elements.

        Returns
        -------
        Tensor
            Tensor containing the value of sech(x) for each element.
        """
        return self.cosh() ** -1

    def abs(self) -> Tensor:
        """Absolute values of tensor elements.

        Returns
        -------
            Tensor
            Tensor containing the absolute value for each element.
        """
        r = get_engine(self.device).abs(self.data)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def sqrt(self) -> Tensor:
        """Square root of tensor elements.

        Returns
        -------
            Tensor
            Tensor containing the square root value for each element.
        """
        r = get_engine(self.device).sqrt(self.data)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def item(self) -> float | int:
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
        r = self.data.reshape(*shape)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def pad(self, pad_width: int | tuple[int, int] | tuple[tuple[int, int]]) -> Tensor:
        """Returns a padded tensor using zero padding.

        Parameters
        ----------
        pad_width : int | tuple[int, int] | tuple[tuple[int, int]]
            Padding width.
            `int`: Same padding for before and after in all dimensions.
            `tuple[int, int]`: Specific before and after padding in all dimensions.
            `tuple[tuple[int, int]`: Specific before and after padding for each dimension.

        Returns
        -------
        Tensor
            Padded tensor.
        """
        r = get_engine(self.device).pad(self.data, pad_width)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def flatten(self) -> Tensor:
        """Returns a flattened, one-dimensional tensor.

        Returns
        -------
        Tensor
            Flattened, one-dimensional version of the tensor.
        """
        r = self.data.reshape((-1,))
        return Tensor(r, dtype=r.dtype, device=self.device)

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
        r = get_engine(self.device).transpose(self.data, axes=axis)
        return Tensor(r, dtype=r.dtype, device=self.device)

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

    def float(self) -> Tensor:
        """Returns a copy of the tensor with float values.

        Returns
        -------
        Tensor
            Float tensor.
        """
        return self.astype("float32")

    def int(self) -> Tensor:
        """Returns a copy of the tensor with int values.

        Returns
        -------
        Tensor
            Int tensor.
        """
        return self.astype("int32")

    def complex(self) -> Tensor:
        """Returns a copy of the tensor with complex values.

        Returns
        -------
        Tensor
            Complex tensor.
        """
        return self.astype("complex64")

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
        r = get_engine(self.device).append(self.data, values.data, axis=axis)
        return Tensor(r, dtype=r.dtype, device=self.device)

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
        r = get_engine(self.device).flip(self.data, axis=axis)
        return Tensor(r, dtype=r.dtype, device=self.device)

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
        r = get_engine(self.device).moveaxis(self.data, from_axis, to_axis)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def squeeze(self) -> Tensor:
        """Removes axis with length one from the tensor."""
        r = self.data.squeeze()
        return Tensor(r, dtype=r.dtype, device=self.device)

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
        r = get_engine(self.device).argmax(self.data, axis=axis)
        return Tensor(r, dtype=r.dtype, device=self.device)

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
        r = get_engine(self.device).resize(self.data, shape)
        return Tensor(r, dtype=r.dtype, device=self.device)

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
        r = self.data.repeat(n_repeats, axis)
        return Tensor(r, dtype=r.dtype, device=self.device)

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
        r = get_engine(self.device).clip(self.data, min_value, max_value)
        return Tensor(r, dtype=r.dtype, device=self.device)

    def copy(self) -> Tensor:
        """Creates a copy of the tensor."""
        t = Tensor(self.data, dtype=self.dtype, copy=True, device=self.device)
        t.grad = self.grad if self.grad is None else self.grad.copy()
        return t
