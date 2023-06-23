"""tensor module"""

import pandas as pd
import numpy as np


class Tensor():
    """Tensor base class.
    
    array_like: Data to initialize the tensor [optional].
    """

    def __init__(self, array_like=None):
        if array_like is not None:
            if isinstance(array_like, np.ndarray):
                self.data = array_like.astype('float32')
            elif isinstance(array_like, (list, int, float)):
                self.data = np.array(array_like).astype('float32')
            else:
                raise ValueError('array_like must be np.ndarray, list, int or float')
        else:
            self._data = None

        self.grad = None
        self.params = {}


    def __repr__(self):
        return self._data.__repr__().replace('array', 'tnsor')

    def __call__(self):
        return self.data

    def __getitem__(self, item):
        return Tensor(self.data[item])

    @property
    def data(self):
        """Gets data value."""
        return self._data

    @data.setter
    def data(self, other):
        """Sets data value"""
        self._data = other
        self.shape = self._data.shape
        self.ndim = self._data.ndim
        self.len = len(self._data) if self.ndim > 0 else 0
        self.T = self._data.T


    def sum(self, axis=None, keepdims=False):
        """Computes the sum along a given axis."""
        return Tensor(np.sum(self._data, axis=axis, keepdims=keepdims))

    def mean(self, axis=None, keepdims=False):
        """Computes the mean along a given axis."""
        return Tensor(np.mean(self._data, axis=axis, keepdims=keepdims))

    def var(self, axis=None, keepdims=False):
        """Computes the variance along a given axis."""
        return Tensor(np.var(self._data, axis=axis, ddof=1, keepdims=keepdims))

    def std(self, axis=None, keepdims=False):
        """Computes the standard deviation along a given axis."""
        return Tensor(np.std(self._data, axis=axis, keepdims=keepdims))

    def exp(self):
        """Computes the expoential for each value."""
        return Tensor(np.exp(self._data))

    def log(self):
        """Computes the logarithm for each value."""
        return Tensor(np.log(self._data))

    def tanh(self):
        """Computes the hyperbolic tangent for each value."""
        return Tensor(np.tanh(self._data))

    def min(self, axis=None, keepdims=False):
        """Computes the hyperbolic tangent for each value."""
        return Tensor(self._data.min(axis=axis, keepdims=keepdims))

    def max(self, axis=None, keepdims=False):
        """Computes the hyperbolic tangent for each value."""
        return Tensor(self._data.max(axis=axis, keepdims=keepdims))

    def item(self):
        """Gets the values of a tensor."""
        return self._data.item()

    def reshape(self, shape):
        """Returns a reshaped tensor."""
        return Tensor(self._data.reshape(*shape))

    def round(self, decimals):
        """Returns a tensor with rounded values."""
        return Tensor(self._data.round(decimals))

    # operator overloading
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

    # helpers

    def __tensorify(self, other):
        if not isinstance(other, Tensor):
            return Tensor(other)
        return other


def pd_to_tensor(dataframe: pd.DataFrame) -> Tensor:
    """Converts a Pandas DataFrame to a tensor.

    Args:
        dataframe: Pandas DataFrame object.
    
    Returns:
        Tensor."""
    return Tensor(dataframe.to_numpy())

def expand_dims(tensor: Tensor, axis=0) -> Tensor:
    """Extends the dimension of a tensor.

    Args:
        tensor: Tensor to be extended.
        axis: Axis where a dimension is added.
    
    Returns:
        Tensor with extended dimensions.
    """
    return Tensor(np.expand_dims(tensor.data, axis=axis))

def match_dims(tensor: Tensor, dims: int) -> Tensor:
    """Extends the dimension of a tensor.

    Args:
        tensor: Tensor to be extended.
        axis: Axis where a dimension is added.
    
    Returns:
        Tensor with extended dimensions.
    """
    while tensor.ndim < dims:
        tensor = expand_dims(tensor, axis=-1)

    return tensor

def zeros(shape) -> Tensor:
    """Creates a tensor of a given shape with all values being zero.

    Args:
        shape: Shape of the tensor.

    Returns:
        Tensor with all values being zero.
    """
    return Tensor(np.zeros(shape))

def ones(shape) -> Tensor:
    """Creates a tensor of a given shape with all values being one.

    Args:
        shape: Shape of the tensor.

    Returns:
        Tensor with all values being one.
    """
    return Tensor(np.ones(shape))

def zeros_like(tensor: Tensor) -> Tensor:
    """Creates a tensor of shape of a given other tensor with all values being zero.

    Args:
        tensor: Tensor whose shape is to be used.

    Returns:
        Tensor with all values being zero.
    """
    return Tensor(np.zeros_like(tensor.data))

def ones_like(tensor: Tensor) -> Tensor:
    """Creates a tensor of shape of a given other tensor with all values being one.

    Args:
        tensor: Tensor whose shape is to be used.

    Returns:
        Tensor with all values being one.
    """
    return Tensor(np.ones_like(tensor.data))

def randn(shape) -> Tensor:
    """Creates a tensor of a given shape with random values according to a normal distribution.

    Args:
        shape: Shape of the tensor.

    Returns:
        Tensor with random values.
    """
    return Tensor(np.random.randn(*shape))

def randint(lower_bound, upper_bound, shape) -> Tensor:
    """Creates a tensor of a given shape with random integer values.

    Args:
        lower_bound: Lower bound of int values.
        upper_bound: Upper bound of int values.
        shape: Shape of the tensor.

    Returns:
        Tensor with random values.
    """
    return Tensor(np.random.randint(lower_bound, upper_bound, shape))

def shuffle(tensor1: Tensor, tensor2: Tensor,
            batch_size: int = None) -> (Tensor|Tensor):
    """Shuffles two tensors of equal size along a axis 0 equally.

    Args:
        x: First tensors to be shuffled.
        y: Second tensors to be shuffled.
        batch_size: Number of samples to be returned [optional].
    
    Returns:
        t1_shuffled: First shuffled tensor.
        t2_shuffled: Second shuffled tensor.

    Raises:
        Error: If tensors are not of equal size along a axis 0.
    """
    if tensor1.len != tensor2.len:
        raise Exception(f'Tensors must have equal lengths along axis 0')

    length = tensor1.len
    shuffle_index = np.arange(length)
    batch_size = batch_size if batch_size else length
    np.random.shuffle(shuffle_index)
    t1_shuffled = tensor1[shuffle_index]
    t2_shuffled = tensor2[shuffle_index]
    return t1_shuffled[:batch_size], t2_shuffled[:batch_size]
