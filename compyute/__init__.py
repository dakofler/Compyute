"""Compyute module"""

from . import _version, engine, nn, preprocessing, random
from .base_tensor import *
from .dtypes import Dtype
from .engine import Device
from .tensor_functions import *

__version__ = _version.__version__

cpu = Device.CPU
cuda = Device.CUDA
int32 = Dtype.INT32
int64 = Dtype.INT64
float16 = Dtype.FLOAT16
float32 = Dtype.FLOAT32
float64 = Dtype.FLOAT64
complex64 = Dtype.COMPLEX64
complex128 = Dtype.COMPLEX128
