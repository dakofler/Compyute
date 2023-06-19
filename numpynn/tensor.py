"""tensor base class"""

import numpy as np


class Tensor():
    """Tensor base class.
    
    array_like: Data to initialize the tensor [optional].
    shape: if no array is provided, a zero tensor can be initialized [optional].
    """

    def __init__(self, array_like=None):
        if array_like is not None:
            if isinstance(array_like, np.ndarray):
                self._data = array_like.astype('float32')
            else:
                self._data = np.array(array_like).astype('float32')

            self.shape = self._data.shape
            self.ndim = self._data.ndim
        else:
            self._data = None

        self.grad = None
        self.delta = None
        self.mmtm = None
        self.velo = None

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

    # operator overloading

    def __repr__(self):
        return self._data.__repr__().replace('array', 'tnsor')

    def __add__(self, other):
        other = self.__tensorify(other)
        return self._data + other.data

    def __mul__(self, other):
        other = self.__tensorify(other)
        return self._data * other.data

    def __sub__(self, other):
        other = self.__tensorify(other)
        return self._data - other.data

    def __truediv__(self, other):
        other = self.__tensorify(other)
        return self._data / other.data

    def __floordiv__(self, other):
        other = self.__tensorify(other)
        return self._data // other.data

    def __pow__(self, other: int):
        return self._data ** other

    def __mod__(self, other: int):
        return self._data % other

    def __matmul__(self, other: int):
        other = self.__tensorify(other)
        return self._data @ other

    def __tensorify(self, other):
        if not isinstance(other, Tensor):
            return Tensor(other)
        return other
