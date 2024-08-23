"""
Compyute is a toolbox for building and training and analyzing neural networks
only using NumPy/CuPy under the hood to perform computations.
"""

import pathlib

from . import backend, nn, preprocessing, random
from .backend import *
from .base_tensor import *
from .tensor_ops import *
from .typing import *
from .utils import *

__version__ = pathlib.Path(f"{pathlib.Path(__file__).parent}/VERSION").read_text(
    encoding="utf-8"
)
