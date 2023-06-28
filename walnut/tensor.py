"""Tensor module"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd
import numpy as np
import numpy.typing as npt


ShapeLike = tuple[int, ...]
NumpyArray = npt.NDArray[Any]


@dataclass(init=False, repr=False)
class Tensor:
    """Tensor object."""

    def __init__(self, data: NumpyArray | list[Any] | float | int | None = None):
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
    def data(self, other: NumpyArray | list[Any] | float | int | None):
        if other is None:
            return
        if isinstance(other, np.ndarray):
            self._data = other.astype("float32")
        elif isinstance(other, (list, float, int)):
            self._data = np.array(other).astype("float32")
        else:
            raise ValueError("data must be NumpyArray, list, int or float")

        self.shape = self._data.shape
        self.ndim = self._data.ndim
        self.len = len(self._data) if self._data.ndim > 0 else 0
        self.T = self._data.T

    def __repr__(self) -> str:
        return self._data.__repr__().replace("array", "tnsor")

    def __call__(self) -> NumpyArray:
        return self.data

    def __getitem__(self, item) -> Tensor:
        return Tensor(self.data[item])

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


def pd_to_tensor(df: pd.DataFrame) -> Tensor:
    """Converts a Pandas DataFrame into a Tensor.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame object to convert.

    Returns
    -------
    Tensor
        Tensor object.
    """
    return Tensor(df.to_numpy())


def expand_dims(x: Tensor, axis: ShapeLike) -> Tensor:
    """Extends the dimensions of a tensor.

    Parameters
    ----------
    x : Tensor
        Tensor whose dimensions are to be extended.
    axis : AxisLike]
        Where to insert the new dimension.

    Returns
    -------
    Tensor
        Tensor with extended dimensions.
    """
    return Tensor(np.expand_dims(x.data, axis=axis))


def match_dims(x: Tensor, dims: int) -> Tensor:
    """Extends the dimensions of a tensor to fit a given number of dims.

    Parameters
    ----------
    x : Tensor
        Tensor to be extended.
    dims : int
        Number of dimensions needed.

    Returns
    -------
    Tensor
        Tensor with extended dimensions.
    """
    while x.ndim < dims:
        x = expand_dims(x, axis=(-1,))

    return x


def zeros(shape: ShapeLike) -> Tensor:
    """Creates a tensor of a given shape with all values being zero.

    Parameters
    ----------
    ShapeLike
        Shape of the new tensor.

    Returns
    -------
    Tensor
        Tensor with all values being zero.
    """
    return Tensor(np.zeros(shape))


def ones(shape: ShapeLike) -> Tensor:
    """Creates a tensor of a given shape with all values being one.

    Parameters
    ----------
    ShapeLike
        Shape of the new tensor.

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    return Tensor(np.ones(shape))


def zeros_like(x: Tensor) -> Tensor:
    """Creates a tensor based on the shape of a given other tensor with all values being zero.

    Parameters
    ----------
    x : Tensor
        Tensor whose shape is used.

    Returns
    -------
    Tensor
        Tensor with all values being zero.
    """
    return Tensor(np.zeros_like(x.data))


def ones_like(x: Tensor) -> Tensor:
    """Creates a tensor based on the shape of a given other tensor with all values being one.

    Parameters
    ----------
    x : Tensor
        Tensor whose shape is used.

    Returns
    -------
    Tensor
        Tensor with all values being one.
    """
    return Tensor(np.ones_like(x.data))


def randn(shape: ShapeLike) -> Tensor:
    """Creates a tensor of a given shape with random values following a normal distribution.

    Parameters
    ----------
    ShapeLike
        Shape of the new tensor.

    Returns
    -------
    Tensor
        Tensor with random values.
    """
    return Tensor(np.random.randn(*shape))


def randint(lower_bound: int, upper_bound: int, shape: ShapeLike) -> Tensor:
    """Creates a tensor of a given shape with random integer values.

    Parameters
    ----------
    lower_bound : int
        Lower bound for random values.
    upper_bound : int
        Upper bound for random values.
    ShapeLike
        Shape of the new tensor.

    Returns
    -------
    Tensor
        Tensor with random values.
    """
    return Tensor(np.random.randint(lower_bound, upper_bound, shape))


def shuffle(
    x1: Tensor, x2: Tensor, batch_size: int | None = None
) -> tuple[Tensor, Tensor]:
    """Shuffles two tensors equally along axis 0.

    Parameters
    ----------
    x1 : Tensor
        First tensor to be shuffled.
    x2 : Tensor
        Second tensor to be shuffled.
    batch_size : int | None, optional
        Number of samples to be returned, by default None.
        If None, all samples are returned.

    Returns
    -------
    tuple[Tensor, Tensor]
        Shuffled tensors.

    Raises
    ------
    ValueError
        If tensors are not of equal size along a axis 0
    """
    if x1.len != x2.len:
        raise ValueError("Tensors must have equal lengths along axis 0")

    length = x1.len
    shuffle_index = np.arange(length)
    batch_size = batch_size if batch_size else length
    np.random.shuffle(shuffle_index)
    y1 = x1[shuffle_index]
    y2 = x2[shuffle_index]
    return y1[:batch_size], y2[:batch_size]
