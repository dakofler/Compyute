"""Tensor module"""

from __future__ import annotations
from types import ModuleType
from typing import Optional, Sequence
import numpy
from .engine import (
    check_device_availability,
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

__all__ = ["tensor", "Tensor"]


class ShapeError(Exception):
    """Incompatible tensor shapes."""


def tensor(
    data: ArrayLike | ScalarLike,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DtypeLike] = None,
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
    device = infer_device(data) if device is None else device
    data = get_engine(device).array(data, copy=copy, dtype=dtype)
    return Tensor(data, requires_grad)


class Tensor:
    """Tensor object."""

    def __init__(
        self,
        data: ArrayLike,
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
    def data(self) -> ArrayLike:
        """Tensor data."""
        return self.__data

    @data.setter
    def data(self, value: ArrayLike | ScalarLike) -> None:

        # TODO: Check if value is array or scalar

        self.__data = value

    @property
    def __e(self) -> ModuleType:
        return get_engine(self.device)

    @property
    def device(self) -> DeviceLike:
        """Device the tensor is stored on."""
        return infer_device(self.__data)

    def to_device(self, device: DeviceLike) -> None:
        """Moves the tensor to a specified device."""
        if self.device == device:
            return
        check_device_availability(device)

        if device == "cuda":
            self.__data = numpy_to_cupy(self.__data)
        else:
            self.__data = cupy_to_numpy(self.__data)

        if self.grad is not None:
            self.grad.to_device(device)

    @property
    def dtype(self) -> DtypeLike:
        """Tensor datatype."""
        return str(self.__data.dtype)

    @property
    def ndim(self) -> int:
        """Tensor dimensions."""
        return self.__data.ndim

    @property
    def size(self) -> int:
        """Tensor size."""
        return self.__data.size

    @property
    def shape(self) -> ShapeLike:
        """Tensor shape."""
        return self.__data.shape

    @property
    def T(self) -> Tensor:
        """Tensor transposed."""
        return self.transpose()

    # ----------------------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        prefix = "Tensor("
        dtype = self.dtype
        shape = self.shape
        device = self.device
        array_string = numpy.array2string(
            self.cpu().data,
            max_line_width=100,
            prefix=prefix,
            separator=", ",
            precision=4,
            floatmode="fixed",
        )
        return f"{prefix}{array_string}, {dtype=}, {shape=}, {device=})"

    def __getitem__(self, idx: Tensor | ArrayLike | int) -> Tensor:
        if isinstance(idx, Tensor):
            i = idx.data
        elif isinstance(idx, tuple):
            i = tuple(j.data if isinstance(j, Tensor) else j for j in idx)
        else:
            i = idx
        item = self.__data[i]
        return Tensor(item) if isinstance(item, ArrayLike) else item

    def __setitem__(self, idx, value) -> None:
        idx = idx.data if isinstance(idx, Tensor) else idx
        value = value.data if isinstance(value, Tensor) else value
        self.__data[idx] = value

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
        return Tensor(self.__data + self.__to_array(other))

    def __radd__(self, other: ScalarLike) -> Tensor:
        return self + other

    def __mul__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.__data * self.__to_array(other))

    def __rmul__(self, other: ScalarLike) -> Tensor:
        return self * other

    def __pow__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.__data ** self.__to_array(other))

    def __rpow__(self, other: ScalarLike) -> Tensor:
        return tensor(other, self.device) ** self

    def __neg__(self) -> Tensor:
        return self * -1

    def __sub__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.__data - self.__to_array(other))

    def __rsub__(self, other: ScalarLike) -> Tensor:
        return -self + other

    def __truediv__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.__data / self.__to_array(other))

    def __rtruediv__(self, other: ScalarLike) -> Tensor:
        return self**-1 * other

    def __floordiv__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.__data // self.__to_array(other))

    def __rfloordiv__(self, other: Tensor | ScalarLike) -> Tensor:
        return (other / self).int().astype(self.dtype)

    def __mod__(self, other: int) -> Tensor:
        return Tensor(self.__data % other)

    def __matmul__(self, other: Tensor) -> Tensor:
        return Tensor(self.__data @ other.data)

    def __lt__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.__data < self.__to_array(other))

    def __gt__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.__data > self.__to_array(other))

    def __le__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.__data <= self.__to_array(other))

    def __ge__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.__data >= self.__to_array(other))

    def __eq__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.__data == self.__to_array(other))

    def __ne__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.__data != self.__to_array(other))

    def __iadd__(self, other: Tensor | ScalarLike) -> Tensor:
        self.__data += self.__to_array(other)
        return self

    def __isub__(self, other: Tensor | ScalarLike) -> Tensor:
        self.__data -= self.__to_array(other)
        return self

    def __imul__(self, other: Tensor | ScalarLike) -> Tensor:
        self.__data *= self.__to_array(other)
        return self

    def __idiv__(self, other: Tensor | ScalarLike) -> Tensor:
        self.__data /= self.__to_array(other)
        return self

    def __ifloordiv__(self, other: Tensor | ScalarLike) -> Tensor:
        self.__data //= self.__to_array(other)
        return self

    def __imod__(self, other: Tensor | ScalarLike) -> Tensor:
        self.__data %= self.__to_array(other)
        return self

    def __ipow__(self, other: Tensor | ScalarLike) -> Tensor:
        self.__data **= self.__to_array(other)
        return self

    def __len__(self) -> int:
        return self.shape[0]

    def __hash__(self):
        return id(self)

    def __array__(self, dtype=None, copy=None):
        return self.astype(dtype).to_numpy()

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
        return tensor(self.__data, self.device, dtype=dtype)

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
            Reshaped tensor.
        """
        return Tensor(self.__data.reshape(*shape))

    def flatten(self) -> Tensor:
        """Returns a flattened, one-dimensional tensor."""
        return Tensor(self.__data.reshape((-1,)))

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
        return self.moveaxis(from_axis=axes[0], to_axis=axes[1])

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
        return Tensor(self.__e.expand_dims(self.__data, axis=axis))

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
        return Tensor(self.__e.resize(self.__data, shape))

    def repeat(self, n_repeats: int, axis: AxisLike) -> Tensor:
        """Repeat elements of a tensor.

        Parameters
        ----------
        n_repeats : int
            Number of repeats.
        axis : AxisLike
            Axis, along which the values are repeated.

        Returns
        -------
        Tensor
            Tensor with repeated values.
        """
        return Tensor(self.__data.repeat(n_repeats, axis))

    def tile(self, n_repeats: int, axis: AxisLike) -> Tensor:
        """Repeat elements of a tensor.

        Parameters
        ----------
        n_repeats : int
            Number of repeats.
        axis : AxisLike
            Axis, along which the values are repeated.

        Returns
        -------
        Tensor
            Tensor with repeated values.
        """
        repeats = [1] * self.ndim
        repeats[axis] = n_repeats
        return Tensor(self.__e.tile(self.__data, tuple(repeats)))

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
        return Tensor(self.__e.pad(self.__data, padding))

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
        padding = tuple((int(0), shape[i] - self.shape[i]) for i in range(self.ndim))
        return self.pad(padding)

    def moveaxis(self, from_axis: AxisLike, to_axis: AxisLike) -> Tensor:
        """Move axes of an array to new positions. Other axes remain in their original order.

        Parameters
        ----------
        from_axis : AxisLike
            Original positions of the axes to move. These must be unique.
        to_axis : AxisLike
            Destination positions for each of the original axes. These must also be unique.

        Returns
        -------
        Tensor
            Tensor with moved axes.
        """
        return Tensor(self.__e.moveaxis(self.__data, from_axis, to_axis))

    def squeeze(self) -> Tensor:
        """Removes axis with length one from the tensor."""
        return Tensor(self.__data.squeeze())

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
        return Tensor(self.__e.flip(self.__data, axis=axis))

    # ----------------------------------------------------------------------------------------------
    # MEMORY/DEVICE METHODS
    # ----------------------------------------------------------------------------------------------

    def copy(self) -> Tensor:
        """Creates a copy of the tensor."""
        t = tensor(self.__data, self.device, self.dtype, copy=True)
        t.grad = None if self.grad is None else self.grad.copy()
        t.requires_grad = self.requires_grad
        return t

    def item(self) -> ScalarLike:
        """Returns the scalar value of the tensor data."""
        return self.__data.item()

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

    def sum(self, axis: Optional[AxisLike] = None, keepdims: bool = False) -> Tensor:
        """Sum of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike, optional
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
        return Tensor(self.__data.sum(axis=axis, keepdims=keepdims))

    def prod(self, axis: Optional[AxisLike] = None, keepdims: bool = False) -> Tensor:
        """Product of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike, optional
            Axis over which the product is computed, by default None.
            If none it is computed over the flattened tensor.
        keepdims : bool, optional
            Whether to keep the tensors dimensions, by default False.
            If false the tensor is collapsed along the given axis.

        Returns
        -------
        Tensor
            Tensor containing the product of elements.
        """
        return Tensor(self.__data.prod(axis=axis, keepdims=keepdims))

    def mean(self, axis: Optional[AxisLike] = None, keepdims: bool = False) -> Tensor:
        """Mean of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike, optional
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
        return Tensor(self.__data.mean(axis=axis, keepdims=keepdims))

    def var(self, axis: Optional[AxisLike] = None, ddof: int = 0, keepdims: bool = False) -> Tensor:
        """Variance of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike, optional
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
        return Tensor(self.__data.var(axis=axis, ddof=ddof, keepdims=keepdims))

    def std(self, axis: Optional[AxisLike] = None, keepdims: bool = False) -> Tensor:
        """Standard deviation of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike, optional
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
        return Tensor(self.__data.std(axis=axis, keepdims=keepdims))

    def min(self, axis: Optional[AxisLike] = None, keepdims: bool = False) -> Tensor:
        """Minimum of tensor elements over a given axis.

        Parameters
        ----------
        axis : AxisLike, optional
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
        return Tensor(self.__data.min(axis=axis, keepdims=keepdims))

    def max(self, axis: Optional[AxisLike] = None, keepdims: bool = False) -> Tensor:
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
        return Tensor(self.__data.max(axis=axis, keepdims=keepdims))

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
        return Tensor(self.__data.round(decimals))

    def exp(self) -> Tensor:
        """Exponential of tensor element."""
        return Tensor(self.__e.exp(self.__data))

    def log(self) -> Tensor:
        """Natural logarithm of tensor elements."""
        return Tensor(self.__e.log(self.__data))

    def log10(self) -> Tensor:
        """Logarithm with base 10 of tensor elements."""
        return Tensor(self.__e.log10(self.__data))

    def log2(self) -> Tensor:
        """Logarithm with base 2 of tensor elements."""
        return Tensor(self.__e.log2(self.__data))

    def sin(self) -> Tensor:
        """Sine of tensor elements."""
        return Tensor(self.__e.sin(self.__data))

    def sinh(self) -> Tensor:
        """Hyperbolic sine of tensor elements."""
        return Tensor(self.__e.sinh(self.__data))

    def cos(self) -> Tensor:
        """Cosine of tensor elements."""
        return Tensor(self.__e.cos(self.__data))

    def cosh(self) -> Tensor:
        """Hyperbolic cosine of tensor elements."""
        return Tensor(self.__e.cosh(self.__data))

    def tan(self) -> Tensor:
        """Tangent of tensor elements."""
        return Tensor(self.__e.tan(self.__data))

    def tanh(self) -> Tensor:
        """Hyperbolical tangent of tensor elements."""
        return Tensor(self.__e.tanh(self.__data))

    def sech(self) -> Tensor:
        """Hyperbolic secant of tensor elements."""
        return self.cosh() ** -1

    def abs(self) -> Tensor:
        """Absolute values of tensor elements."""
        return Tensor(self.__e.abs(self.__data))

    def sqrt(self) -> Tensor:
        """Square root of tensor elements."""
        return Tensor(self.__e.sqrt(self.__data))

    def fft1d(
        self,
        n: Optional[int] = None,
        axis: AxisLike = -1,
        dtype: Optional[ComplexLike] = None,
    ) -> Tensor:
        """Computes the 1D Fast Fourier Transform over a specified axis.

        Parameters
        ----------
        n : int, optional
            Length of the transformed axis of the output, by default None.
        axis : AxisLike, optional
            Axis over which to compute the FFT, by default -1.
        dtype : ComplexLike, optional
            Datatype of the output tensor, by default None.

        Returns
        -------
        Tensor
            Complex tensor.
        """
        return tensor(
            self.__e.fft.fft(self.__data, n=n, axis=axis), device=self.device, dtype=dtype
        )

    def ifft1d(
        self,
        n: Optional[int] = None,
        axis: AxisLike = -1,
        dtype: Optional[ComplexLike] = None,
    ) -> Tensor:
        """Computes the inverse 1D Fast Fourier Transform over a specified axis.

        Parameters
        ----------
        n : int, optional
            Length of the transformed axis of the output, by default None.
        axis : AxisLike, optional
            Axis over which to compute the inverse FFT, by default -1.
        dtype : FloatLike, optional
            Datatype of the output tensor, by default None.

        Returns
        -------
        Tensor
            Float tensor.
        """
        return tensor(
            self.__e.fft.ifft(self.__data, n=n, axis=axis), device=self.device, dtype=dtype
        )

    def fft2d(
        self,
        s: Optional[ShapeLike] = None,
        axes: AxisLike = (-2, -1),
        dtype: Optional[ComplexLike] = None,
    ) -> Tensor:
        """Computes the 2D Fast Fourier Transform over two specified axes.

        Parameters
        ----------
        n : ShapeLike, optional
            Shape (length of each transformed axis) of the output, by default None.
        axes : AxisLike, optional
            Axes over which to compute the FFT, by default (-2, -1).
        dtype : ComplexLike, optional
            Datatype of the output tensor, by default None.

        Returns
        -------
        Tensor
            Complex tensor.
        """
        return tensor(
            self.__e.fft.fft2(self.__data, s=s, axes=axes), device=self.device, dtype=dtype
        )

    def ifft2d(
        self,
        s: Optional[ShapeLike] = None,
        axes: AxisLike = (-2, -1),
        dtype: Optional[ComplexLike] = None,
    ) -> Tensor:
        """Applies the inverse 1D Fast Fourier Transform to the tensor.

        Parameters
        ----------
        n : ShapeLike, optional
            Shape (length of each transformed axis) of the output, by default None.
        axes : AxisLike, optional
            Axes over which to compute the inverse FFT, by default (-2, -1).
        dtype : ComplexLike, optional
            Datatype of the output tensor, by default None.

        Returns
        -------
        Tensor
            Complex tensor.
        """
        return tensor(
            self.__e.fft.ifft2(self.__data, s=s, axes=axes), device=self.device, dtype=dtype
        )

    def real(self, dtype: Optional[DtypeLike] = None) -> Tensor:
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
        return tensor(self.__e.real(self.__data), device=self.device, dtype=dtype)

    def append(self, values: Tensor, axis: AxisLike = -1) -> Tensor:
        """Returns a copy of the tensor with appended values.

        Parameters
        ----------
        values : Tensor
            Values to append.
        axis : AxisLike, optional
            Axis alowng which to append the values, by default -1.

        Returns
        -------
        Tensor
            Tensor containing appended values.
        """
        return Tensor(self.__e.append(self.__data, values.data, axis=axis))

    def argmax(self, axis: Optional[AxisLike] = None) -> Tensor:
        """Returns the indices of maximum values along a given axis.

        Parameters
        ----------
        axis : Optional[AxisLike] = None
            Axis, along which the maximum value is located, by default None.

        Returns
        -------
        Tensor
            Tensor containing indices.
        """
        return Tensor(self.__e.argmax(self.__data, axis=axis))

    def clip(
        self, min_value: Optional[ScalarLike] = None, max_value: Optional[ScalarLike] = None
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
        return Tensor(self.__e.clip(self.__data, min_value, max_value))

    def unique(self) -> Tensor:
        """Returns the unique ordered values of the tensor."""
        return Tensor(self.__e.unique(self.__data))

    def split(self, splits: int | Sequence[int], axis: AxisLike = -1) -> list[Tensor]:
        """Returns a list of new tensors by splitting the tensor.

        Parameters
        ----------
        splits : int | list[int]
            If an int is given, the tensor is split into n equally sized tensors.
            If a list of indices is given, they represent the indices at which to
            split the tensor along the given axis.
        axis : AxisLike, optional
            Axis along which to split the tensor, by default -1.

        Returns
        -------
        list[Tensor]
            List of tensors containing the split data.
        """
        return [Tensor(s) for s in self.__e.split(self.__data, splits, axis=axis)]

    def get_diagonal(self, d: int = 0) -> Tensor:
        """Extract a diagonal or construct a diagonal tensor.

        Parameters
        ----------
        d : int, optional
            Index of the diagonal, by default 0.
            - 0 ... main diagonal
            - >0 ... above main diagonal
            - <0 ... below main diagonal

        Returns
        -------
        Tensor
            The extracted diagonal or constructed diagonal tensor.
        """
        return Tensor(self.__e.diag(self.__data, k=d))

    def tril(self, d: int = 0) -> Tensor:
        """Returns the lower triangle of a tensor below the d-th diagonal of the last two dimensions.

        Parameters
        ----------
        d : int, optional
            Index of the diagonal, by default 0.
            - 0 ... main diagonal
            - >0 ... above main diagonal
            - <0 ... below main diagonal

        Returns
        -------
        Tensor
            Lower triangle tensor.
        """
        return Tensor(self.__e.tril(self.__data, k=d))

    def triu(self, d: int = 0) -> Tensor:
        """Returns the upper triangle of a tensor above the d-th diagonal of the last two dimensions.

        Parameters
        ----------
        d : int, optional
            Index of the diagonal, by default 0.
            - 0 ... main diagonal
            - >0 ... above main diagonal
            - <0 ... below main diagonal

        Returns
        -------
        Tensor
            Upper triangle tensor.
        """
        return Tensor(self.__e.triu(self.__data, k=d))

    def __to_array(self, data: Tensor | ArrayLike | ScalarLike) -> ArrayLike | ScalarLike:
        """Returns array-compatible data."""
        if isinstance(data, Tensor):
            if data.device != self.device:
                raise ValueError("Devices do not match.")
            return data.data
        if isinstance(data, ArrayLike):
            if infer_device(data) != self.device:
                raise ValueError("Devices do not match.")
            return data
        return data
