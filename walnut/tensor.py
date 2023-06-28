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


@dataclass(init=False, repr=False)
class Tensor:
    """Tensor object."""

    def __init__(
        self, data: NumpyArray | list[Any] | float | int | np.float32 | None = None
    ):
        """Tensor object.

        Parameters
        ----------
        data : NumpyArray | list[Any] | float | int | None, optional
            Data to initialize the tensor, by default None.
        """
        self._data: NumpyArray = np.empty(0, dtype="float32")
        self.data = data
        self.grad: NumpyArray = np.empty(0, dtype="float32")
        self.params: dict[str, NumpyArray] = {}
        self.shape: ShapeLike = self._data.shape
        self.ndim: int = self._data.ndim
        self.len: int = len(self._data) if self._data.ndim > 0 else 0
        self.T: NumpyArray = self._data.T

    @property
    def data(self) -> NumpyArray:
        """Tensor data."""
        return self._data

    @data.setter
    def data(self, other: NumpyArray | list[Any] | float | int | np.float32 | None):
        if other is None:
            return
        if isinstance(other, np.ndarray):
            self._data = other.astype("float32")
        elif isinstance(other, (list, float, int, np.ndarray, numpyFloat)):
            self._data = np.array(other).astype("float32")
        else:
            raise ValueError("data must be NumpyArray, list, int or float")

        self.shape = self._data.shape
        self.ndim = self._data.ndim
        self.len = len(self._data) if self._data.ndim > 0 else 0
        self.T = self._data.T

    # function overloading

    def __repr__(self) -> str:
        return self._data.__repr__().replace("array", "tnsor")

    def __call__(self) -> NumpyArray:
        return self.data

    def __getitem__(self, item) -> Tensor:
        return Tensor(self.data[item])

    def __add__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self._data + other.data)

    def __mul__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self._data * other.data)

    def __sub__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self._data - other.data)

    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self._data / other.data)

    def __floordiv__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self._data // other.data)

    def __pow__(self, other: int) -> Tensor:
        return Tensor(self._data**other)

    def __mod__(self, other: int) -> Tensor:
        return Tensor(self._data % other)

    def __matmul__(self, other: Tensor) -> Tensor:
        return Tensor(self._data @ other.data)

    def __lt__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self._data < other.data)

    def __gt__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self._data > other.data)

    def __le__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self._data <= other.data)

    def __ge__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self._data >= other.data)

    def __eq__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self._data == other.data)

    def __ne__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        return Tensor(self._data != other.data)

    def __isub__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self._data += other.data
        return Tensor(self._data)

    def __iadd__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self._data += other.data
        return Tensor(self._data)

    def __imul__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self._data *= other.data
        return Tensor(self._data)

    def __idiv__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self._data /= other.data
        return Tensor(self._data)

    def __ifloordiv__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self._data //= other.data
        return Tensor(self._data)

    def __imod__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self._data %= other.data
        return Tensor(self._data)

    def __ipow__(self, other: Tensor | float | int) -> Tensor:
        other = self.__tensorify(other)
        self._data **= other.data
        return Tensor(self._data)

    def __neg__(self) -> Tensor:
        self._data = -1.0 * self._data
        return Tensor(self._data)

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
        return Tensor(np.sum(self._data, axis=axis, keepdims=keepdims))

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
        return Tensor(np.mean(self._data, axis=axis, keepdims=keepdims))

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
        return Tensor(np.var(self._data, axis=axis, ddof=1, keepdims=keepdims))

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
        return Tensor(np.std(self._data, axis=axis, keepdims=keepdims))

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
        return Tensor(self._data.min(axis=axis, keepdims=keepdims))

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
        return Tensor(self._data.max(axis=axis, keepdims=keepdims))

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
        """Exponential of tensor elements.

        Returns
        -------
        Tensor
            Tensor containing the value of e**x for each element.
        """
        return Tensor(np.exp(self._data))

    def log(self) -> Tensor:
        """Logarithm of tensor elements.

        Returns
        -------
            Tensor
                Tensor containing the value of log(x) for each element.
        """
        return Tensor(np.log(self._data))

    def tanh(self) -> Tensor:
        """Hyperbolical tangent of tensor elements.

        Returns
        -------
            Tensor
            Tensor containing the value of tanh(x) for each element.
        """
        return Tensor(np.tanh(self._data))

    def item(self, *args) -> float:
        """Returns the scalar value of the tensor data.

        Returns
        -------
        float
            Scalar of the tensor data.
        """
        return self._data.item(args)

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
        return Tensor(self._data.reshape(*shape))

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
        return Tensor(np.pad(self._data, paddings))
