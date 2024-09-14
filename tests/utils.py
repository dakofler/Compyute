"""Testing utilities."""

import numpy
import torch

from compyute.backend import Device, cpu
from compyute.nn.parameter import Parameter
from compyute.random.random import seed, uniform, uniform_int
from compyute.tensors import ShapeLike, Tensor
from compyute.typing import float32, int64


@seed(42)
def get_random_floats(
    shape: ShapeLike,
    torch_grad: bool = True,
    device: Device = cpu,
    low: float = -0.1,
    high: float = 0.1,
) -> tuple[Tensor, torch.Tensor]:
    """Returns a compyute tensor and a torch tensor initialized equally."""
    compyute_x = uniform(shape, low, high, device=device, dtype=float32)
    torch_x = torch.tensor(compyute_x.to_numpy())
    if torch_grad:
        torch_x.requires_grad = True
    return compyute_x, torch_x


@seed(42)
def get_random_integers(
    shape: ShapeLike, device: Device = cpu, low: int = 0, high: int = 10
) -> tuple[Tensor, torch.Tensor]:
    """Returns a compyute tensor and a torch tensor initialized equally."""
    compyute_x = uniform_int(shape, low, high, device=device, dtype=int64)
    torch_x = torch.tensor(compyute_x.to_numpy()).long()
    compyute_x.to_device(device)
    return compyute_x, torch_x


@seed(42)
def get_random_params(
    shape: ShapeLike, device: Device = cpu
) -> tuple[Parameter, torch.nn.Parameter]:
    """Returns a compyute tensor and a torch parameter tensor initialized equally."""
    data = uniform(shape, dtype=float32) * 0.1
    compyute_x = Parameter(data)
    torch_x = torch.nn.Parameter(torch.tensor(data.to_numpy()).float())
    compyute_x.to_device(device)
    return compyute_x, torch_x


def is_close(x1: Tensor | Parameter, x2: torch.Tensor, tol: float = 1e-5) -> bool:
    """Checks whether a compyute and torch tensor contain equal values."""
    return numpy.allclose(x1.to_numpy(), x2.detach().numpy(), tol, tol)
