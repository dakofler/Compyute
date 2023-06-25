"""tensor module"""

import pandas as pd
import numpy as np


class Tensor():
    """Tensor base class.
    
    ### Parameters
        data: `ndarray`, `list`, `float` or `int`, optional
            Data to initialize the tensor.
    """

    __slots__ = '_data', 'grad', 'params', 'shape', 'ndim', 'len', 'T'

    def __init__(self, data: np.ndarray | list | float | int = None):
        self.data = data
        self.grad: np.ndarray = None
        self.params: dict[str, np.ndarray] = {}

    @property
    def data(self) -> np.ndarray:
        """Tensor data"""
        return self._data

    @data.setter
    def data(self, other):
        if other is not None:
            if isinstance(other, np.ndarray):
                self._data = other.astype('float32')
            elif isinstance(other, (list, int, float, np.float32)):
                self._data = np.array(other).astype('float32')
            else:
                raise ValueError('data must be np.ndarray, list, int or float')

            self.shape = self._data.shape
            self.ndim = self._data.ndim
            self.len = len(self._data) if self.ndim > 0 else 0
            self.T = self._data.T
        else:
            self._data = None
            self.shape = None
            self.ndim = None
            self.len = None
            self.T = None

    def __repr__(self) -> str:
        return self._data.__repr__().replace('array', 'tnsor')

    def __call__(self) -> np.ndarray:
        return self.data

    def __getitem__(self, item) -> np.ndarray:
        return Tensor(self.data[item])

    def sum(self, axis: int | tuple[int] = None, keepdims: bool = False):
        """Sum of tensor elements over a given axis.

        ### Parameters
            axis: `int` or `tuple[int]`, optional
                Axis over which the sum is computed.
                By default, it is computed over the flattened tensor.
            keepdims: `bool`, optional
                Whether to keep the tensors dimensions.
                By default the tensor is collapsed along the given axis.

        ### Returns
            y: `Tensor`
                Tensor containing the sum of elements.
        """
        return Tensor(np.sum(self._data, axis=axis, keepdims=keepdims))

    def mean(self, axis: int | tuple[int] = None, keepdims: bool = False):
        """Mean of tensor elements over a given axis.

        ### Parameters
            axis: `int` or `tuple[int]`, optional
                Axis over which the mean is computed.
                By default, it is computed over the flattened tensor.
            keepdims: `bool`, optional
                Whether to keep the tensors dimensions.
                By default the tensor is collapsed along the given axis.

        ### Returns
            y: `Tensor`
                Tensor containing the mean of elements.
        """
        return Tensor(np.mean(self._data, axis=axis, keepdims=keepdims))

    def var(self, axis: int | tuple[int] = None, keepdims: bool = False):
        """Variance of tensor elements over a given axis.

        ### Parameters
            axis: `int` or `tuple[int]`, optional
                Axis over which the variance is computed.
                By default, it is computed over the flattened tensor.
            keepdims: `bool`, optional
                Whether to keep the tensors dimensions.
                By default the tensor is collapsed along the given axis.

        ### Returns
            y: `Tensor`
                Tensor containing the variance of elements.
        """
        return Tensor(np.var(self._data, axis=axis, ddof=1, keepdims=keepdims))

    def std(self, axis: int | tuple[int] = None, keepdims: bool = False):
        """Standard deviation of tensor elements over a given axis.

        ### Parameters
            axis: `int` or `tuple[int]`, optional
                Axis over which the standard deviation is computed.
                By default, it is computed over the flattened tensor.
            keepdims: `bool`, optional
                Whether to keep the tensors dimensions.
                By default the tensor is collapsed along the given axis.

        ### Returns
            y: `Tensor`
                Tensor containing the standard deviation of elements.
        """
        return Tensor(np.std(self._data, axis=axis, keepdims=keepdims))

    def min(self, axis: int | tuple[int] = None, keepdims: bool = False):
        """Minimum of tensor elements over a given axis.

        ### Parameters
            axis: `int` or `tuple[int]`, optional
                Axis over which the minimum is taken.
                By default, it is computed over the flattened tensor.
            keepdims: `bool`, optional
                Whether to keep the tensors dimensions.
                By default the tensor is collapsed along the given axis.

        ### Returns
            y: `Tensor`
                Tensor containing the minimum of elements.
        """
        return Tensor(self._data.min(axis=axis, keepdims=keepdims))

    def max(self, axis: int | tuple[int] = None, keepdims: bool = False):
        """Maximum of tensor elements over a given axis.

        ### Parameters
            axis: `int` or `tuple[int]`, optional
                Axis over which the maximum is taken.
                By default, it is computed over the flattened tensor.
            keepdims: `bool`, optional
                Whether to keep the tensors dimensions.
                By default the tensor is collapsed along the given axis.

        ### Returns
            y: `Tensor`
                Tensor containing the maximum of elements.
        """
        return Tensor(self._data.max(axis=axis, keepdims=keepdims))

    def round(self, decimals: int):
        """Rounds the value of tensor elements.

        ### Parameters
            decimals: `int`
                Decimal places of rounded values.

        ### Returns
            y: `Tensor`
                Tensor containing the rounded values.
        """
        return Tensor(self._data.round(decimals))

    def exp(self):
        """Exponential of tensor elements.

        ### Returns
            y: `Tensor`
                Tensor containing the value of e**x for each element.
        """
        return Tensor(np.exp(self._data))

    def log(self):
        """Logarithm of tensor elements.

        ### Returns
            y: `Tensor`
                Tensor containing the value of log(x) for each element.
        """
        return Tensor(np.log(self._data))

    def tanh(self):
        """Hyperbolical tangent of tensor elements.

        ### Returns
            y: `Tensor`
                Tensor containing the value of tanh(x) for each element.
        """
        return Tensor(np.tanh(self._data))

    def item(self, *args):
        """Returns the scalar value of the tensor data.

        ### Parameters
            *args:
                `None`: Only works for tensors with one element.\n
                `int`: Index of the value in the flattened tensor.\n
                `tuple[int]`: Index of the value

        ### Returns
            item: `float`
                Scalar of the tensor data"""
        return self._data.item(args)

    def reshape(self, shape):
        """Returns a reshaped tensor of data to fit a given shape.

        ### Parameters
            shape: `tuple[int]`
                Shape of the new tensor.

        ### Returns
            y: `Tensor`
                Reshapded tensor.
        """
        return Tensor(self._data.reshape(*shape))

    def pad(self, widths: tuple[int]):
        """Returns a padded tensor using zero padding.

        ### Parameters
            widths : `tuple[int]`
                Padding width for each dimension of the tensor.

        ### Returns
            y: `Tensor`
                Padded tensor.
        """
        paddings = tuple((w, w) for w in widths)
        return Tensor(np.pad(self._data, paddings))

    def __add__(self, other):
        other = self.__tensorify(other)
        return Tensor(self._data + other.data)

    def __mul__(self, other):
        other = self.__tensorify(other)
        return Tensor(self._data * other.data)

    def __sub__(self, other):
        other = self.__tensorify(other)
        return Tensor(self._data - other.data)

    def __truediv__(self, other):
        other = self.__tensorify(other)
        return Tensor(self._data / other.data)

    def __floordiv__(self, other):
        other = self.__tensorify(other)
        return Tensor(self._data // other.data)

    def __pow__(self, other: int):
        return Tensor(self._data ** other)

    def __mod__(self, other: int):
        return Tensor(self._data % other)

    def __matmul__(self, other):
        other = self.__tensorify(other)
        return Tensor(self._data @ other.data)

    def __lt__(self, other):
        other = self.__tensorify(other)
        return Tensor(self._data < other.data)

    def __gt__(self, other):
        other = self.__tensorify(other)
        return Tensor(self._data > other.data)

    def __le__(self, other):
        other = self.__tensorify(other)
        return Tensor(self._data <= other.data)

    def __ge__(self, other):
        other = self.__tensorify(other)
        return Tensor(self._data >= other.data)

    def __eq__(self, other):
        other = self.__tensorify(other)
        return Tensor(self._data == other.data)

    def __ne__(self, other):
        other = self.__tensorify(other)
        return Tensor(self._data != other.data)

    def __isub__(self, other):
        other = self.__tensorify(other)
        self._data += other.data
        return Tensor(self._data)

    def __iadd__(self, other):
        other = self.__tensorify(other)
        self._data += other.data
        return Tensor(self._data)

    def __imul__(self, other):
        other = self.__tensorify(other)
        self._data *= other.data
        return Tensor(self._data)

    def __idiv__(self, other):
        other = self.__tensorify(other)
        self._data /= other.data
        return Tensor(self._data)

    def __ifloordiv__(self, other):
        other = self.__tensorify(other)
        self._data //= other.data
        return Tensor(self._data)

    def __imod__(self, other):
        other = self.__tensorify(other)
        self._data %= other.data
        return Tensor(self._data)

    def __ipow__(self, other):
        other = self.__tensorify(other)
        self._data **= other.data
        return Tensor(self._data)

    def __neg__(self):
        self._data = -1.0 * self._data
        return Tensor(self._data)

    def __tensorify(self, other):
        if not isinstance(other, Tensor):
            return Tensor(other)
        return other


def pd_to_tensor(df: pd.DataFrame) -> Tensor:
    """Converts a Pandas DataFrame into a Tensor.

    ### Parameters
        df: `DataFrame`
            Pandas DataFrame object to convert.
    
    ### Returns
        y: `Tensor`
            Tensor object."""
    return Tensor(df.to_numpy())

def expand_dims(x: Tensor, axis: int | tuple[int]) -> Tensor:
    """Extends the dimensions of a tensor.

    ### Parameters
        x: `Tensor`
            Tensor whose dimensions are to be extended.
        axis: `int` or `tuple[int]`
            Where to insert the new dimension.
    
    ### Returns
        y: `Tensor`
            Tensor with extended dimensions.
    """
    return Tensor(np.expand_dims(x.data, axis=axis))

def match_dims(x: Tensor, dims: int) -> Tensor:
    """Extends the dimensions of a tensor to fit a given number of dims.

    ### Parameters
        x: `Tensor`
            Tensor to be extended.
        dims: `int`
            Number of dimensions needed.
    
    ### Returns
        y: `Tensor`
            Tensor with extended dimensions.
    """
    while x.ndim < dims:
        x = expand_dims(x, axis=-1)

    return x

def zeros(shape: tuple[int]) -> Tensor:
    """Creates a tensor of a given shape with all values being zero.

    ### Parameters
        shape: `tuple[int]`
            Shape of the new tensor.

    ### Returns
        y: `Tensor`
            Tensor with all values being zero.
    """
    return Tensor(np.zeros(shape))

def ones(shape: tuple[int]) -> Tensor:
    """Creates a tensor of a given shape with all values being one.

    ### Parameters
        shape: `tuple[int]`
            Shape of the new tensor.

    ### Returns
        y: `Tensor`
            Tensor with all values being one.
    """
    return Tensor(np.ones(shape))

def zeros_like(x: Tensor) -> Tensor:
    """Creates a tensor based on the shape of a given other tensor with all values being zero.

    ### Parameters
        x: `Tensor`
            Tensor whose shape is used.

    ### Returns
        y: `Tensor`
            Tensor with all values being zero.
    """
    return Tensor(np.zeros_like(x.data))

def ones_like(x: Tensor) -> Tensor:
    """Creates a tensor based on the shape of a given other tensor with all values being one.

    ### Parameters
        x: `Tensor`
            Tensor whose shape is used.

    ### Returns
        y: `Tensor`
            Tensor with all values being one.
    """
    return Tensor(np.ones_like(x.data))

def randn(shape: tuple[int]) -> Tensor:
    """Creates a tensor of a given shape with random values following a normal distribution.

    ### Parameters
        shape: `tuple[int]`
            Shape of the new tensor.

    ### Returns
        y: `Tensor`
            Tensor with random values.
    """
    return Tensor(np.random.randn(*shape))

def randint(lower_bound: int, upper_bound: int, shape: tuple[int]) -> Tensor:
    """Creates a tensor of a given shape with random integer values.

    ### Parameters
        lower_bound: `int`
            Lower bound for random values.
        upper_bound: `int`
            Upper bound for random values.
        shape: `tuple[int]`
            Shape of the new tensor.

    ### Returns
        y: `Tensor`
            Tensor with random values.
    """
    return Tensor(np.random.randint(lower_bound, upper_bound, shape))

def shuffle(x1: Tensor, x2: Tensor,
            batch_size: int = None) -> (Tensor|Tensor):
    """Shuffles two tensors equally along axis 0.

    ### Parameters
        x1: `Tensor`
            First tensor to be shuffled.
        x2: `Tensor`
            Second tensor to be shuffled.
        batch_size: `int`, optional
            Number of samples to be returned.By default all samples are returned.
    
    ### Returns
        y1: `Tensor`
            First shuffled tensor.
        y2: `Tensor`
            Second shuffled tensor.

    ### Raises
        ValueError:
            If tensors are not of equal size along a axis 0.
    """
    if x1.len != x2.len:
        raise ValueError('Tensors must have equal lengths along axis 0')

    length = x1.len
    shuffle_index = np.arange(length)
    batch_size = batch_size if batch_size else length
    np.random.shuffle(shuffle_index)
    y1 = x1[shuffle_index]
    y2 = x2[shuffle_index]
    return y1[:batch_size], y2[:batch_size]
