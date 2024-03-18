"""Test utils module"""

import numpy
import cupy
import torch

from compyute.engine import cupy_to_numpy
from compyute.nn.parameter import Parameter
from compyute.random import uniform_int, set_seed, uniform
from compyute.tensor import Tensor
from compyute.types import ShapeLike, ArrayLike


set_seed(42)


def get_vals_float(
    shape: ShapeLike,
    torch_grad: bool = True,
    device: str = "cpu",
) -> tuple[Tensor, torch.Tensor]:
    """Returns a compyute tensor and a torch tensor initialized equally."""
    compyute_x = uniform(shape, dtype="float32")
    torch_x = torch.from_numpy(compyute_x.to_numpy())
    if torch_grad:
        torch_x.requires_grad = True
    compyute_x.to_device(device)
    return compyute_x, torch_x


def get_vals_int(
    shape: ShapeLike, device: str = "cpu", low: int = 0, high: int = 10
) -> tuple[Tensor, torch.Tensor]:
    """Returns a compyute tensor and a torch tensor initialized equally."""
    compyute_x = uniform_int(shape, low=low, high=high, dtype="int64")
    torch_x = torch.from_numpy(compyute_x.to_numpy()).long()
    compyute_x.to_device(device)
    return compyute_x, torch_x


def get_params(
    shape: ShapeLike, device: str = "cpu"
) -> tuple[Parameter, torch.nn.Parameter]:
    """Returns a compyute tensor and a torch parameter tensor initialized equally."""
    data = uniform(shape) * 0.1
    compyute_x = Parameter(data, dtype="float32")
    torch_x = torch.nn.Parameter(torch.from_numpy(data.to_numpy()).float())
    compyute_x.to_device(device)
    return compyute_x, torch_x


def validate(
    x1: Tensor | Parameter, x2: torch.Tensor | None, tol: float = 1e-5
) -> bool:
    """Checks whether a compyute and torch tensor contain equal values."""
    return numpy.allclose(x1.to_numpy(), x2.detach().numpy(), tol, tol)
