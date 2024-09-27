"""Tensor reshaping operations."""

from typing import Optional

from ..tensors import DimLike, ShapeLike, Tensor
from .creation_ops import identity

__all__ = [
    "diagonal",
    "reshape",
    "flatten",
    "transpose",
    "insert_dim",
    "repeat",
    "tile",
    "pad",
    "pad_to_shape",
    "movedim",
    "squeeze",
    "flip",
    "broadcast_to",
    "pooling1d",
    "pooling2d",
]


def diagonal(x: Tensor) -> Tensor:
    """Expands a tensor by turning the last dim into a diagonal matrix."""
    return x.view((*x.shape, 1)) * identity(x.shape[-1])


def reshape(x: Tensor, shape: ShapeLike) -> Tensor:
    """Returns a new view of the tensor of a given shape.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    shape : ShapeLike
        Shape of the new tensor.

    Returns
    -------
    Tensor
        Reshaped tensor.
    """
    return Tensor(x.data.reshape(*shape))


def flatten(x: Tensor) -> Tensor:
    """Returns a flattened 1D-tensor.

    Parameters
    ----------
    x: Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Flattened tensor.
    """
    return x.view((-1,))


def transpose(x: Tensor, dims: Optional[tuple[int, ...]] = None) -> Tensor:
    """Transposes a tensor by swapping two dimensions.

    Parameters
    ----------
    x: Tensor
        Input tensor.
    dims : tuple[int, int], optional
        Permutation of output dimensions. Defaults to ``None``.
        If ``None`` dimensions are reversed.

    Returns
    -------
    Tensor
        Transposed tensor.
    """
    return x.transpose(dims)


def insert_dim(x: Tensor, dim: int) -> Tensor:
    """Returns a view of the tensor with an added dimension.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : int
        Where to insert the new dimension.

    Returns
    -------
    Tensor
        Tensor with an added dimension.
    """
    if dim == -1:
        return x.view((*x.shape, 1))
    if dim < 0:
        dim += 1
    return x.view((*x.shape[:dim], 1, *x.shape[dim:]))


def repeat(x: Tensor, n_repeats: int, dim: int) -> Tensor:
    """Repeat elements of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    n_repeats : int
        Number of repeats.
    dim : int
        Dimension along which the values are repeated.

    Returns
    -------
    Tensor
        Tensor with repeated values.
    """
    return Tensor(x.data.repeat(n_repeats, dim))


def tile(x: Tensor, n_repeats: int, dim: int) -> Tensor:
    """Repeat elements of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    n_repeats : int
        Number of repeats.
    dim : int
        Dimension along which the values are repeated.

    Returns
    -------
    Tensor
        Tensor with repeated values.
    """
    repeats = [1] * x.ndim
    repeats[dim] = n_repeats
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
        | ``int``: same padding width at the begining and end of all dimensions.
        | ``tuple[int, int]``: specific widths at the beginning and end of all dimensions.
        | ``tuple[tuple[int, int]]``: specific widths in the beginning and end for each dim.

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
    if x.shape == shape:
        return x
    padding = tuple((int(0), shape[i] - x.shape[i]) for i in range(x.ndim))
    return pad(x, padding)


def movedim(x: Tensor, from_dim: int, to_dim: int) -> Tensor:
    """Move dimensions of a tensor to new positions.
    Other dimensions remain in their original order.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    from_dim : int
        Original position of the dimension to move.
    to_dim : int
        New position of the dimension.

    Returns
    -------
    Tensor
        Tensor with a moved dimensions.
    """
    return Tensor(x.device.module.moveaxis(x.data, from_dim, to_dim))


def squeeze(x: Tensor) -> Tensor:
    """Returns a tensor with removed dimensions where ``dim=1``.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor with removed dimensions.
    """
    return x.squeeze()


def flip(x: Tensor, dim: Optional[DimLike] = None) -> Tensor:
    """Returns a tensor with flipped elements along given dim.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : DimLike, optional
        | Dimension along which to flip the tensor. Defaults to ``None``.
        | ``None``: all dimensions are flipped
        | ``int`` or ``tuple[int, ...]``: specified dimensions are flipped.

    Returns
    -------
    Tensor
        Tensor containing flipped values.
    """
    return Tensor(x.device.module.flip(x.data, dim))


def broadcast_to(x: Tensor, shape: ShapeLike) -> Tensor:
    """Broadcast a tensor to a new shape.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    shape : ShapeLike
        Shape of the new tensor

    Returns
    -------
    Tensor
        Broadcasted tensor.
    """
    return Tensor(x.device.module.broadcast_to(x.data, shape))


def pooling1d(x: Tensor, window_size: int, stride: int = 1):
    """Returns a windowed view of a tensor across the last dim.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    window_size : int
        Size of the pooling window.
    stride : int
        Stride of the pooling operation.

    Returns
    -------
    Tensor
        Windowed view of the input tensor.
    """

    # compute output shape
    out = (x.shape[-1] - window_size) // stride + 1
    out_shape = (*x.shape[:-1], out, window_size)

    # compute output strides
    x_str = x.strides
    out_strides = (*x_str[:-1], x_str[-1] * stride, x_str[-1])

    str_func = x.device.module.lib.stride_tricks.as_strided
    return Tensor(str_func(x.data, out_shape, out_strides))


def pooling2d(x: Tensor, window_size: int, stride: int = 1):
    """Returns a windowed view of a tensor across the last two dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    window_size : int
        Size of the pooling window.
    stride : int
        Stride of the pooling operation.

    Returns
    -------
    Tensor
        Windowed view of the input tensor.
    """

    # compute output shape
    out = (x.shape[-1] - window_size) // stride + 1
    out_shape = (*x.shape[:-2], out, out, window_size, window_size)

    # compute output strides
    x_str = x.strides
    out_strides = (*x_str[:-2], x_str[-2] * stride, x_str[-1] * stride, *x_str[-2:])

    str_func = x.device.module.lib.stride_tricks.as_strided
    return Tensor(str_func(x.data, out_shape, out_strides))
