"""Tensor module"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import numpy.typing as npt


ShapeLike = tuple[int, ...]
NumpyArray = npt.NDArray[Any]
numpyFloat = np.float32 | np.float64


class ShapeError(Exception):
    """Incompatible tensor shapes."""


@dataclass(init=False, repr=False, slots=True)
class Tensor:
    """Tensor object."""

    data: NumpyArray
    grad: NumpyArray
    params: dict[str, NumpyArray]

    def __init__(
        self,
        values: NumpyArray | list[Any] | float | int | np.float32 | None = None,
        dtype: str = "float32",
    ):
        """Tensor object.

        Parameters
        ----------
        values : NumpyArray | list[Any] | float | int | None, optional
            Data to initialize the tensor, by default None.
        """
        if values is None:
            self.data = np.empty(0, dtype=dtype)
        elif isinstance(values, np.ndarray):
            self.data = values.astype(dtype, copy=values.dtype != dtype)
        elif isinstance(values, (list, float, int, numpyFloat)):
            self.data = np.array(values).astype(dtype)
        else:
            raise ValueError("values must be NumpyArray, list, int or float")

        self.grad: NumpyArray = np.empty(0, dtype="float32")
        self.params: dict[str, NumpyArray] = {}

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
    def T(self) -> NumpyArray:
        """Tensor data transposed."""
        return self.data.T

    @property
    def dtype(self) -> str:
        """Tensor data datatype."""
        return str(self.data.dtype)

    # function overloading

    def __repr__(self) -> str:
        return self.data.__repr__().replace("array", "tnsor")

    def __call__(self) -> NumpyArray:
        return self.data

    def __getitem__(self, item) -> Tensor:
        return Tensor(self.data[item])

    def __add__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data + other.data)

    def __mul__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data * other.data)

    def __sub__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data - other.data)

    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data / other.data)

    def __floordiv__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data // other.data)

    def __pow__(self, other: int) -> Tensor:
        return Tensor(self.data**other)

    def __mod__(self, other: int) -> Tensor:
        return Tensor(self.data % other)

    def __matmul__(self, other: Tensor) -> Tensor:
        return Tensor(self.data @ other.data)

    def __lt__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data < other.data)

    def __gt__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data > other.data)

    def __le__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data <= other.data)

    def __ge__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data >= other.data)

    def __eq__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data == other.data)

    def __ne__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self.data != other.data)

    def __isub__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data += other.data
        return Tensor(self.data)

    def __iadd__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data += other.data
        return Tensor(self.data)

    def __imul__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data *= other.data
        return Tensor(self.data)

    def __idiv__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data /= other.data
        return Tensor(self.data)

    def __ifloordiv__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data //= other.data
        return Tensor(self.data)

    def __imod__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data %= other.data
        return Tensor(self.data)

    def __ipow__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self.data **= other.data
        return Tensor(self.data)

    def __neg__(self) -> Tensor:
        self.data = -1.0 * self.data
        return Tensor(self.data)

    def __tensorify(self, other: Tensor | float | int) -> Tensor:
        if not isinstance(other, Tensor):
            return Tensor(other)
        return other

    # functions

    def sum(self, axis: ShapeLike | None = None, keepdims: bool = False) -> Tensor:
        """Sum of tensor elements over a given axis.

        Parameters
        ----------
        axis : ShapeLike | None, optional
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
        return Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))

    def mean(self, axis: ShapeLike | None = None, keepdims: bool = False) -> Tensor:
        """Mean of tensor elements over a given axis.

        Parameters
        ----------
        axis : ShapeLike | None, optional
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
        return Tensor(np.mean(self.data, axis=axis, keepdims=keepdims))

    def var(self, axis: ShapeLike | None = None, keepdims: bool = False) -> Tensor:
        """Variance of tensor elements over a given axis.

        Parameters
        ----------
        axis : ShapeLike | None, optional
            Axis over which the variance is computed, by default None.
            If none it is computed over the flattened tensor.
        keepdims : bool, optional
            Whether to keep the tensors dimensions, by default False.
            If false the tensor is collapsed along the given axis.

        Returns
        -------
        Tensor
            Tensor containing the variance of elements.
        """
        return Tensor(np.var(self.data, axis=axis, ddof=1, keepdims=keepdims))

    def std(self, axis: ShapeLike | None = None, keepdims: bool = False) -> Tensor:
        """Standard deviation of tensor elements over a given axis.

        Parameters
        ----------
        axis : ShapeLike | None, optional
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
        return Tensor(np.std(self.data, axis=axis, keepdims=keepdims))

    def min(self, axis: ShapeLike | None = None, keepdims: bool = False) -> Tensor:
        """Minimum of tensor elements over a given axis.

        Parameters
        ----------
        axis : ShapeLike | None, optional
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

    def max(self, axis: ShapeLike | None = None, keepdims: bool = False) -> Tensor:
        """Maximum of tensor elements over a given axis.

        Parameters
        ----------
        axis : ShapeLike | None, optional
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
        """Exponential of tensor elements.

        Returns
        -------
        Tensor
            Tensor containing the value of e**x for each element.
        """
        return Tensor(np.exp(self.data))

    def log(self) -> Tensor:
        """Logarithm of tensor elements.

        Returns
        -------
            Tensor
                Tensor containing the value of log(x) for each element.
        """
        return Tensor(np.log(self.data))

    def tanh(self) -> Tensor:
        """Hyperbolical tangent of tensor elements.

        Returns
        -------
            Tensor
            Tensor containing the value of tanh(x) for each element.
        """
        return Tensor(np.tanh(self.data))

    def abs(self) -> Tensor:
        """Absolute values of tensor elements.

        Returns
        -------
            Tensor
            Tensor containing the absolute value for each element.
        """
        return Tensor(np.abs(self.data))

    def item(self, *args) -> float:
        """Returns the scalar value of the tensor data.

        Returns
        -------
        float
            Scalar of the tensor data.
        """
        return self.data.item(args)

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
        return Tensor(self.data.reshape(*shape))

    def pad(self, widths: tuple[int, ...]) -> Tensor:
        """Returns a padded tensor using zero padding.

        Parameters
        ----------
        widths : tuple[int, ...]
            Padding width for each dimension of the tensor.

        Returns
        -------
        Tensor
            Padded tensor.
        """
        paddings = tuple((w, w) for w in widths)
        return Tensor(np.pad(self.data, paddings))

    def reset_params(self, reset_data: bool = False):
        """Resets additional parameters to improve memory usage.

        Parameters
        ----------
        reset_data : bool, optional
            Whether to also reset the tensor data, by default False.
        """
        if reset_data:
            self.data = np.empty(0, dtype=self.dtype)
        self.grad = np.empty(0, dtype="float32")
        self.params: dict[str, NumpyArray] = {}

    def flatten(self) -> Tensor:
        """Returns a flattened, one-dimensional tensor.

        Returns
        -------
        Tensor
            Flattened, one-dimensional version of the tensor.
        """
        return Tensor(self.data.reshape((-1,)))

    def transpose(self, axis: ShapeLike) -> Tensor:
        """Transposes a tensor along given axes.

        Parameters
        ----------
        axes : ShapeLike
            Permutation of axes of the transposed tensor.

        Returns
        -------
        Tensor
            Transposed tensor.
        """
        return Tensor(np.transpose(self.data, axes=axis))
