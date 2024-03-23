"""Tensor module"""

from __future__ import annotations
from types import ModuleType
import numpy
from .engine import (
    check_device,
    cupy_to_numpy,
    get_engine,
    infer_device,
    numpy_to_cupy,
)
from .types import (
    ArrayLike,
    AxisLike,
    ComplexLike,
    DeviceLike,
    DtypeLike,
    ScalarLike,
    ShapeLike,
)

__all__ = ["Tensor"]


class ShapeError(Exception):
    """Incompatible tensor shapes."""


class Tensor:
    """Tensor object."""

    def __init__(
        self,
        data: ArrayLike | ScalarLike,
        dtype: DtypeLike | None = None,
        copy: bool = False,
        device: DeviceLike | None = None,
    ) -> None:
        """Tensor object.

        Parameters
        ----------
        data : ArrayLike | ScalarLike
            Data to initialize the tensor.
        dtype: DtypeLike, optional
            Datatype of the tensor data, by default None. If None, the dtype is inferred.
        copy: bool, optional
            If true, the data object is copied (may impact performance), by default False.
        device: DeviceLike, optional
            Device the tensor is stored on ("cuda" or "cpu"), by default "cpu".
        """

        self.__device = infer_device(data) if device is None else device
        self.data = self.__engine.array(data, copy=copy, dtype=dtype)
        self.grad: Tensor | None = None
        self.__iterator = 0

    # ----------------------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------------------

    @property
    def __engine(self) -> ModuleType:
        return get_engine(self.__device)

    @property
    def device(self) -> DeviceLike:
        """Device the tensor is stored on."""
        return self.__device

    @property
    def dtype(self) -> str:
        """Tensor data datatype."""
        return str(self.data.dtype)

    @property
    def ndim(self) -> int:
        """Tensor dimensions."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Tensor size."""
        return self.data.size

    @property
    def shape(self) -> ShapeLike:
        """Tensor shape."""
        return self.data.shape

    @property
    def T(self) -> ArrayLike:
        """Tensor data transposed."""
        return Tensor(self.data.T)

    # ----------------------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        prefix = "Tensor("
        dtype = self.dtype
        shape = self.shape
        device = self.__device
        array_string = numpy.array2string(
            self.cpu().data,
            max_line_width=100,
            prefix=prefix,
            separator=", ",
            precision=4,
        )
        return f"{prefix}{array_string}, {dtype=}, {shape=}, {device=})"

    def __call__(self) -> ArrayLike:
        return self.data

    def __getitem__(self, idx: Tensor | ArrayLike | int) -> Tensor:
        if isinstance(idx, Tensor):
            i = idx.data
        elif isinstance(idx, tuple):
            i = tuple(j.data if isinstance(j, Tensor) else j for j in idx)
        else:
            i = idx
        item = self.data[i]
        return Tensor(item) if isinstance(item, ArrayLike) else item

    def __setitem__(self, idx, value) -> None:
        idx = idx.data if isinstance(idx, Tensor) else idx
        value = value.data if isinstance(value, Tensor) else value
        self.data[idx] = value

    def __iter__(self) -> Tensor:
        self.__iterator = 0
        return self

    def __next__(self) -> Tensor | ScalarLike:
        if self.__iterator < self.shape[0]:
            data = self[self.__iterator]
            self.__iterator += 1
            return data
        raise StopIteration

    def __add__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data + parse_data(other))

    def __radd__(self, other: ScalarLike) -> Tensor:
        return self + other

    def __mul__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data * parse_data(other))

    def __rmul__(self, other: ScalarLike) -> Tensor:
        return self * other

    def __pow__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data ** parse_data(other))

    def __rpow__(self, other: ScalarLike) -> Tensor:
        return Tensor(other, device=self.__device) ** self

    def __neg__(self) -> Tensor:
        return self * -1

    def __sub__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data - parse_data(other))

    def __rsub__(self, other: ScalarLike) -> Tensor:
        return -self + other

    def __truediv__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data / parse_data(other))

    def __rtruediv__(self, other: ScalarLike) -> Tensor:
        return self**-1 * other

    def __floordiv__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data // parse_data(other))

    def __rfloordiv__(self, other: Tensor | ScalarLike) -> Tensor:
        return (other / self).int().astype(self.dtype)

    def __mod__(self, other: int) -> Tensor:
        return Tensor(self.data % other)

    def __matmul__(self, other: Tensor) -> Tensor:
        return Tensor(self.data @ other.data)

    def __lt__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data < parse_data(other))

    def __gt__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data > parse_data(other))

    def __le__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data <= parse_data(other))

    def __ge__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data >= parse_data(other))

    def __eq__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data == parse_data(other))

    def __ne__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data != parse_data(other))

    def __isub__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data += parse_data(other)
        return self

    def __iadd__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data += parse_data(other)
        return self

    def __imul__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data *= parse_data(other)
        return self

    def __idiv__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data /= parse_data(other)
        return self

    def __ifloordiv__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data //= parse_data(other)
        return self

    def __imod__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data %= parse_data(other)
        return self

    def __ipow__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data **= parse_data(other)
        return self

    def __len__(self) -> int:
        return self.shape[0]

    # ----------------------------------------------------------------------------------------------
    # DTYPE CONVERSIONS
    # ----------------------------------------------------------------------------------------------

    def astype(self, dtype: DtypeLike) -> Tensor:
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
        return Tensor(self.data, dtype=dtype, copy=False)

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

    def reshape(self, shape: ShapeLike) -> Tensor:
        """Returns a view of the tensor tensor of a given shape.

        Parameters
        ----------
        ShapeLike
            Shape of the new tensor.

        Returns
        -------
        Tensor
            Reshapded tensor.
        """
        return Tensor(self.data.reshape(*shape))

    def flatten(self) -> Tensor:
        """Returns a flattened, one-dimensional tensor."""
        return Tensor(self.data.reshape((-1,)))

    def transpose(self, axes: AxisLike = (-2, -1)) -> Tensor:
        """Transposes a tensor by swapping two axes.

        Parameters
        ----------
        axes : AxisLike, optional
            Transpose axes, by default (-2, -1).

        Returns
        -------
        Tensor
            Transposed tensor.
        """
        return self.moveaxis(*axes)

    def insert_dim(self, axis: AxisLike) -> Tensor:
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
        return Tensor(self.__engine.expand_dims(self.data, axis=axis))

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
        return Tensor(self.__engine.resize(self.data, shape))

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
        return Tensor(self.data.repeat(n_repeats, axis))

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
        return Tensor(self.__engine.pad(self.data, pad_width))

    def pad_to_shape(self, shape: ShapeLike) -> Tensor:
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
        pad_width = [(0, shape[i] - self.shape[i]) for i in range(self.ndim)]
        return self.pad(tuple(pad_width))

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
        return Tensor(self.__engine.moveaxis(self.data, from_axis, to_axis))

    def squeeze(self) -> Tensor:
        """Removes axis with length one from the tensor."""
        return Tensor(self.data.squeeze())

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
        return Tensor(self.__engine.flip(self.data, axis=axis))

    # ----------------------------------------------------------------------------------------------
    # MEMORY/DEVICE METHODS
    # ----------------------------------------------------------------------------------------------

    def copy(self) -> Tensor:
        """Creates a copy of the tensor."""
        t = Tensor(self.data, copy=True)
        t.grad = None if self.grad is None else self.grad.copy()
        return t

    def item(self) -> ScalarLike:
        """Returns the scalar value of the tensor data."""
        return self.data.item()

    def to_device(self, device: DeviceLike) -> None:
        """Moves the tensor to a specified device.

        Parameters
        ----------
        device : DeviceLike
            Device the tensor is stored on ("cuda" or "cpu").

        Raises
        ----------
        AttributeErrors
            If device is not "cpu" or "cuda".

        """
        if self.__device == device:
            return

        check_device(device)
        self.__device = device

        if device == "cuda":
            self.data = numpy_to_cupy(self.data)
        else:
            self.data = cupy_to_numpy(self.data)

        if self.grad is not None:
            self.grad.to_device(device)

    def cpu(self):
        """Returns a copy of the tensor on the cpu."""
        if self.__device == "cpu":
            return self

        self.to_device("cpu")
        return self

    def cuda(self):
        """Returns a copy of the tensor on the gpu."""
        if self.__device == "cuda":
            return self

        self.to_device("cuda")
        return self

    # ----------------------------------------------------------------------------------------------
    # OTHER OPERATIONS
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
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims))

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
        return Tensor(self.data.mean(axis=axis, keepdims=keepdims))

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
        return Tensor(self.data.var(axis=axis, ddof=ddof, keepdims=keepdims))

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
        return Tensor(self.data.std(axis=axis, keepdims=keepdims))

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
        return Tensor(self.data.min(axis=axis, keepdims=keepdims))

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
        return Tensor(self.data.max(axis=axis, keepdims=keepdims))

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
        return Tensor(self.data.round(decimals))

    def exp(self) -> Tensor:
        """Exponential of tensor element."""
        return Tensor(self.__engine.exp(self.data))

    def log(self) -> Tensor:
        """Natural logarithm of tensor elements."""
        return Tensor(self.__engine.log(self.data))

    def log10(self) -> Tensor:
        """Logarithm with base 10 of tensor elements."""
        return Tensor(self.__engine.log10(self.data))

    def log2(self) -> Tensor:
        """Logarithm with base 2 of tensor elements."""
        return Tensor(self.__engine.log2(self.data))

    def sin(self) -> Tensor:
        """Sine of tensor elements."""
        return Tensor(self.__engine.sin(self.data))

    def sinh(self) -> Tensor:
        """Hyperbolic sine of tensor elements."""
        return Tensor(self.__engine.sinh(self.data))

    def cos(self) -> Tensor:
        """Cosine of tensor elements."""
        return Tensor(self.__engine.cos(self.data))

    def cosh(self) -> Tensor:
        """Hyperbolic cosine of tensor elements."""
        return Tensor(self.__engine.cosh(self.data))

    def tan(self) -> Tensor:
        """Tangent of tensor elements."""
        return Tensor(self.__engine.tan(self.data))

    def tanh(self) -> Tensor:
        """Hyperbolical tangent of tensor elements."""
        return Tensor(self.__engine.tanh(self.data))

    def sech(self) -> Tensor:
        """Hyperbolic secant of tensor elements."""
        return self.cosh() ** -1

    def abs(self) -> Tensor:
        """Absolute values of tensor elements."""
        return Tensor(self.__engine.abs(self.data))

    def sqrt(self) -> Tensor:
        """Square root of tensor elements."""
        return Tensor(self.__engine.sqrt(self.data))

    def fft1d(
        self,
        n: int | None = None,
        axis: AxisLike = -1,
        dtype: ComplexLike | None = None,
    ) -> Tensor:
        """Computes the 1D Fast Fourier Transform over a specified axis.

        Parameters
        ----------
        n : int | None, optional
            Length of the transformed axis of the output, by default None.
        axis : AxisLike, optional
            Axis over which to compute the FFT, by default -1.
        dtype : ComplexLike | None, optional
            Datatype of the output tensor, by default None.

        Returns
        -------
        Tensor
            Complex tensor.
        """
        return Tensor(self.__engine.fft.fft(self.data, n=n, axis=axis), dtype=dtype)

    def ifft1d(
        self,
        n: int | None = None,
        axis: AxisLike = -1,
        dtype: ComplexLike | None = None,
    ) -> Tensor:
        """Computes the inverse 1D Fast Fourier Transform over a specified axis.

        Parameters
        ----------
        n : int | None, optional
            Length of the transformed axis of the output, by default None.
        axis : AxisLike, optional
            Axis over which to compute the inverse FFT, by default -1.
        dtype : FloatLike | None, optional
            Datatype of the output tensor, by default None.

        Returns
        -------
        Tensor
            Float tensor.
        """
        return Tensor(self.__engine.fft.ifft(self.data, n=n, axis=axis), dtype=dtype)

    def fft2d(
        self,
        s: ShapeLike | None = None,
        axes: AxisLike = (-2, -1),
        dtype: ComplexLike | None = None,
    ) -> Tensor:
        """Computes the 2D Fast Fourier Transform over two specified axes.

        Parameters
        ----------
        n : ShapeLike | None, optional
            Shape (length of each transformed axis) of the output, by default None.
        axes : AxisLike, optional
            Axes over which to compute the FFT, by default (-2, -1).
        dtype : ComplexLike | None, optional
            Datatype of the output tensor, by default None.

        Returns
        -------
        Tensor
            Complex tensor.
        """
        return Tensor(self.__engine.fft.fft2(self.data, s=s, axes=axes), dtype=dtype)

    def ifft2d(
        self,
        s: ShapeLike | None = None,
        axes: AxisLike = (-2, -1),
        dtype: ComplexLike | None = None,
    ) -> Tensor:
        """Applies the inverse 1D Fast Fourier Transform to the tensor.

        Parameters
        ----------
        n : ShapeLike | None, optional
            Shape (length of each transformed axis) of the output, by default None.
        axes : AxisLike, optional
            Axes over which to compute the inverse FFT, by default (-2, -1).
        dtype : ComplexLike | None, optional
            Datatype of the output tensor, by default None.

        Returns
        -------
        Tensor
            Float tensor.
        """
        return Tensor(self.__engine.fft.ifft2(self.data, s=s, axes=axes), dtype=dtype)

    def real(self, dtype: DtypeLike | None = None) -> Tensor:
        """Returns the real parts of the complex tensor.

        Parameters
        ----------
        dtype : DtypeLike | None, optional
            Datatype of the output tensor, by default None.
        """
        return Tensor(self.__engine.real(self.data), dtype=dtype)

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
        return Tensor(self.__engine.append(self.data, values.data, axis=axis))

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
        return Tensor(self.__engine.argmax(self.data, axis=axis))

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
        return Tensor(self.__engine.clip(self.data, min_value, max_value))

    def unique(self) -> Tensor:
        """Returns the unique ordered values of the tensor."""
        return Tensor(self.__engine.unique(self.data))

    def split(self, splits: int | list[int], axis: int = -1) -> list[Tensor]:
        """Returns a list of new tensors by splitting the tensor.

        Parameters
        ----------
        splits : int | list[int]
            If an int is given, the tensor is split into n equally sized tensors.
            If a list of indices is given, they represent the indices at which to
            split the tensor along the given axis.
        axis : int, optional
            Axis along which to split the tensor, by default -1.

        Returns
        -------
        list[Tensor]
            List of tensors containing the split data.
        """
        chunks = self.__engine.split(self.data, splits, axis=axis)
        return [Tensor(c) for c in chunks]


def parse_data(value: Tensor | ScalarLike) -> ArrayLike | ScalarLike:
    if isinstance(value, Tensor):
        return value.data
    return value
