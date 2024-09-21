"""Tensor reshaping operations."""

from typing import Optional

from ..tensors import AxisLike, ShapeLike, Tensor
from .creation_ops import identity

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


def reshape(x: Tensor, shape: ShapeLike) -> Tensor:
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


def transpose(x: Tensor, axes: Optional[AxisLike] = None) -> Tensor:
    """Transposes a tensor by swapping two axes.

    Parameters
    ----------
    x: Tensor
        Input tensor.
    axes : AxisLike, optional
        Permutation of output axes. Defaults to ``None``.
        If ``None`` axes are reversed.

    Returns
    -------
    Tensor
        Transposed tensor.
    """
    return x.transpose(axes)


def insert_dim(x: Tensor, axis: int) -> Tensor:
    """Returns a view of the tensor containing an added dimension at a given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : int
        Where to insert the new dimension.

    Returns
    -------
    Tensor
        Tensor with an added dimension.
    """
    if axis == -1:
        return reshape(x, (*x.shape, 1))
    if axis < 0:
        axis += 1
    return reshape(x, (*x.shape[:axis], 1, *x.shape[axis:]))


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
    return reshape(x, x.shape + (1,) * (target_dims - x.n_axes))


def resize(x: Tensor, shape: ShapeLike) -> Tensor:
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
    return Tensor(x.device.module.resize(x.data, shape))


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
    repeats = [1] * x.n_axes
    repeats[axis] = n_repeats
    return Tensor(x.device.module.tile(x.data, tuple(repeats)))


def pad(
    x: Tensor, padding: int | tuple[int, int] | tuple[tuple[int, int], ...]
) -> Tensor:
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
    return Tensor(x.device.module.pad(x.data, padding))


def pad_to_shape(x: Tensor, shape: ShapeLike) -> Tensor:
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
    padding = tuple((int(0), shape[i] - x.shape[i]) for i in range(x.n_axes))
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
    return Tensor(x.device.module.moveaxis(x.data, from_axis, to_axis))


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
    return x.squeeze()


def flip(x: Tensor, axis: Optional[AxisLike] = None) -> Tensor:
    """Returns a tensor with flipped elements along given axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : AxisLike, optional
        | Axis along which to flip the tensor. Defaults to ``None``.
        | ``None``: all axes are flipped
        | ``int``: only the specified axis is flipped.
        | ``tuple[int, ...]``: all specified axes are flipped.

    Returns
    -------
    Tensor
        Tensor containing flipped values.
    """
    return Tensor(x.device.module.flip(x.data, axis))


def broadcast_to(x: Tensor, shape: ShapeLike) -> Tensor:
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
    return Tensor(x.device.module.broadcast_to(x.data, shape))
