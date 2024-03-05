"""Test utils module"""

import numpy as np
import cupy as cp
import torch

import compyute
from compyute.nn.parameter import Parameter
from compyute.tensor import Tensor, ShapeLike, ArrayLike


compyute.engine.set_seed(42)


def get_vals(
    shape: ShapeLike, torch_grad: bool = True, device: str = "cpu"
) -> tuple[Tensor, torch.Tensor]:
    """Returns a compyute tensor and a torch tensor initialized equally."""
    compyute_x = compyute.random_uniform(shape, dtype="float32")
    torch_x = torch.from_numpy(compyute_x.data)
    if torch_grad:
        torch_x.requires_grad = True
    compyute_x.to_device(device)
    return compyute_x, torch_x


def get_params(
    shape: ShapeLike, T: bool = False, device: str = "cpu"
) -> tuple[Parameter, torch.nn.Parameter]:
    """Returns a compyute tensor and a torch parameter tensor initialized equally."""
    compyute_x = Parameter(compyute.random_uniform(shape), dtype="float32")
    if T:
        torch_x = torch.nn.Parameter(torch.from_numpy(compyute_x.T).float())
    else:
        torch_x = torch.nn.Parameter(torch.from_numpy(compyute_x.data).float())
    compyute_x.to_device(device)
    return compyute_x, torch_x


def validate(
    x1: Tensor | Parameter | ArrayLike, x2: torch.Tensor | None, tol: float = 1e-5
) -> bool:
    """Checks whether a compyute and torch tensor contain equal values."""
    if isinstance(x1, Tensor):
        x1 = x1.data
    if isinstance(x1, cp.ndarray):
        x1 = compyute.engine.cupy_to_numpy(x1)
    return np.allclose(x1, x2.detach().numpy(), tol, tol)
