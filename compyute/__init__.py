"""
Compyute is a toolbox for building and training and analyzing neural networks
only using NumPy/CuPy under the hood to perform computations.
"""

import pathlib

from . import dtypes, engine, nn, preprocessing, random
from .base_tensor import *
from .dtypes import (
    bool,
    complex64,
    complex128,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
)
from .engine import cpu, cuda
from .tensor_functions import *

__version__ = pathlib.Path(f"{pathlib.Path(__file__).parent}/VERSION").read_text(encoding="utf-8")
