"""Tensor reshaping operations."""

from typing import Optional

from ..base_tensor import Tensor, _AxisLike, _ShapeLike
from ..engine import get_engine
from .creating import identity

__all__ = [
    "diagonal",
    "reshape",
    "flatten",
    "transpose",
    "insert_dim",
    "add_dims",
    "resize",
    "repeat",
    "tile",
    "pad",
    "pad_to_shape",
    "moveaxis",
    "squeeze",
    "flip",
    "broadcast_to",
]


def diagonal(x: Tensor) -> Tensor:
    """Expands a tensor by turning the last dim into a diagonal matrix."""
    return insert_dim(x, -1) * identity(x.shape[-1])


def reshape(x: Tensor, shape: _ShapeLike) -> Tensor:
    """Returns a new view of the tensor of a given shape.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    shape : _ShapeLike
        Shape of the new tensor.

    Returns
    -------
    Tensor
        Reshaped tensor.
    """
    return Tensor(x.data.reshape(*shape))


def flatten(x: Tensor) -> Tensor:
    """Returns a flattened, one-dimensional tensor."""
    return reshape(x, shape=((-1,)))


def transpose(x: Tensor, axes: tuple[int, int] = (-2, -1)) -> Tensor:
    """Transposes a tensor by swapping two axes.

    Parameters
    ----------
    x: Tensor
        Input tensor.
    axes : tuple[int, int], optional
        Transpose axes. Defaults to ``(-2, -1)``.

    Returns
    -------
    Tensor
        Transposed tensor.
    """
    if x.ndim < 2:
        return x
    return moveaxis(x, from_axis=axes[0], to_axis=axes[1])


def insert_dim(x: Tensor, axis: _AxisLike) -> Tensor:
    """Returns a view of the tensor containing an added dimension at a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike
        Where to insert the new dimension.

    Returns
    -------
    Tensor
        Tensor with an added dimension.
    """
    return Tensor(get_engine(x.device).expand_dims(x.data, axis=axis))


def add_dims(x: Tensor, target_dims: int) -> Tensor:
    """Returns a view of the tensor with added trailing dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    target_dims : int
        Total number of dimensions needed.

    Returns
    -------
    Tensor
        Tensor with specified number of dimensions.
    """
    return reshape(x, x.shape + (1,) * (target_dims - x.ndim))


def resize(x: Tensor, shape: _ShapeLike) -> Tensor:
    """Returns a new tensor with the specified shape.
    If the new tensor is larger than the original one, it is filled with zeros.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    shape : ShapeLike
        Shape of the new tensor.

    Returns
    -------
    Tensor
        Resized tensor.
    """
    return Tensor(get_engine(x.device).resize(x.data, shape))


def repeat(x: Tensor, n_repeats: int, axis: int) -> Tensor:
    """Repeat elements of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    n_repeats : int
        Number of repeats.
    axis : int
        Axis, along which the values are repeated.

    Returns
    -------
    Tensor
        Tensor with repeated values.
    """
    return Tensor(x.data.repeat(n_repeats, axis))


def tile(x: Tensor, n_repeats: int, axis: int) -> Tensor:
    """Repeat elements of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    n_repeats : int
        Number of repeats.
    axis : int
        Axis, along which the values are repeated.

    Returns
    -------
    Tensor
        Tensor with repeated values.
    """
    repeats = [1] * x.ndim
    repeats[axis] = n_repeats
    return Tensor(get_engine(x.device).tile(x.data, tuple(repeats)))


def pad(x: Tensor, padding: int | tuple[int, int] | tuple[tuple[int, int], ...]) -> Tensor:
    """Returns a padded tensor using zero padding.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    pad_width : int | tuple[int, int] | tuple[tuple[int, int], ...]
        | Padding width(s).
        | ``int``: same padding width at the begining and end of all axes.
        | ``tuple[int, int]``: specific widths at the beginning and end of all axes.
        | ``tuple[tuple[int, int]]``: specific widths in the beginning and end for each axis.

    Returns
    -------
    Tensor
        Padded tensor.
    """
    return Tensor(get_engine(x.device).pad(x.data, padding))


def pad_to_shape(x: Tensor, shape: _ShapeLike) -> Tensor:
    """Returns a padded tensor using zero padding that matches a specified shape.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    shape : ShapeLike
        Final shape of the padded tensor.

    Returns
    -------
    Tensor
        Padded tensor.
    """
    padding = tuple((int(0), shape[i] - x.shape[i]) for i in range(x.ndim))
    return pad(x, padding)


def moveaxis(x: Tensor, from_axis: int, to_axis: int) -> Tensor:
    """Move axes of an array to new positions. Other axes remain in their original order.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    from_axis : int
        Original position of the axis to move.
    to_axis : int
        New position of the axis.

    Returns
    -------
    Tensor
        Tensor with a moved axes.
    """
    return Tensor(get_engine(x.device).moveaxis(x.data, from_axis, to_axis))


def squeeze(x: Tensor) -> Tensor:
    """Returns a tensor with removed axes where ``dim=1``.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor with removed axes.
    """
    return Tensor(x.data.squeeze())


def flip(x: Tensor, axis: Optional[_AxisLike] = None) -> Tensor:
    """Returns a tensor with flipped elements along given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        | Axis alown which to flip the tensor. Defaults to ``None``.
        | ``None``: all axes are flipped
        | ``int``: only the specified axis is flipped.
        | ``tuple[int, ...]``: all specified axes are flipped.

    Returns
    -------
    Tensor
        Tensor containing flipped values.
    """
    return Tensor(get_engine(x.device).flip(x.data, axis=axis))


def broadcast_to(x: Tensor, shape: _ShapeLike) -> Tensor:
    """Broadcast a tensor to a new shape.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    shape : _ShapeLike
        Shape of the new tensor

    Returns
    -------
    Tensor
        Broadcasted tensor.
    """
    return Tensor(get_engine(x.device).broadcast_to(x.data, shape=shape))
