"""Test utils module"""

import numpy as np
import cupy as cp
import torch

import walnut
from walnut.nn.parameter import Parameter
from walnut.tensor import Tensor, ShapeLike, ArrayLike


def get_vals(
    shape: ShapeLike, torch_grad: bool = True, device: str = "cpu"
) -> tuple[Tensor, torch.Tensor]:
    """Returns a walnut tensor and a torch tensor initialized equally."""
    walnut_x = walnut.randn(shape, dtype="float32")
    torch_x = torch.from_numpy(walnut_x.data)
    if torch_grad:
        torch_x.requires_grad = True
    walnut_x.to_device(device)
    return walnut_x, torch_x


def get_params(
    shape: ShapeLike, T: bool = False, device: str = "cpu"
) -> tuple[Parameter, torch.nn.Parameter]:
    """Returns a walnut tensor and a torch parameter tensor initialized equally."""
    walnut_x = Parameter(walnut.randn(shape), dtype="float32")
    if T:
        torch_x = torch.nn.Parameter(torch.from_numpy(walnut_x.T).float())
    else:
        torch_x = torch.nn.Parameter(torch.from_numpy(walnut_x.data).float())
    walnut_x.to_device(device)
    return walnut_x, torch_x


def validate(
    x1: Tensor | Parameter | ArrayLike, x2: torch.Tensor | None, tol: float = 1e-5
) -> bool:
    """Checks whether a walnut and torch tensor contain equal values."""
    if isinstance(x1, Tensor):
        x1 = x1.data
    if isinstance(x1, cp.ndarray):
        x1 = walnut.cuda.cupy_to_numpy(x1)
    return np.allclose(x1, x2.detach().numpy(), tol, tol)
