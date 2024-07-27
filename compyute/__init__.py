"""Compyute."""

import pathlib

from . import engine, nn, preprocessing, random
from .base_tensor import *
from .dtypes import *
from .engine import *
from .tensor_functions import *

__version__ = pathlib.Path(f"{pathlib.Path(__file__).parent}/VERSION").read_text(encoding="utf-8")
