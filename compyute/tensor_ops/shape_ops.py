"""Tensor reshaping operations."""

from collections.abc import Sequence
from typing import Optional

from ..tensors import DimLike, ShapeLike, Tensor
from .creation_ops import identity

__all__ = [
    "append",
    "broadcast_to",
    "concat",
    "diagonal",
    "flatten",
    "flip",
    "insert_dim",
    "movedim",
    "pad",
    "pad_to_shape",
    "pooling1d",
    "pooling2d",
    "repeat1d",
    "repeat2d",
    "reshape",
    "squeeze",
    "split",
    "stack",
    "tile",
    "transpose",
]


def append(x: Tensor, values: Tensor, dim: int = -1) -> Tensor:
    """Returns a copy of the tensor with appended values.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    values : Tensor
        Values to append.
    dim : int, optional
        Dimension along which to append the values. Defaults to ``-1``.

    Returns
    -------
    Tensor
        Tensor containing appended values.
    """
    return Tensor(x.device.module.append(x.data, values.data, axis=dim))


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


def concat(tensors: Sequence[Tensor], dim: DimLike = -1) -> Tensor:
    """Returns a new tensor by joining a sequence of tensors along a given dimension.

    Parameters
    ----------
    tensors : Sequence[Tensor]
        Sequence of Tensors to be joined.
    dim : DimLike, optional
        Dimension along which to join the tensors. Defaults to ``-1``.

    Returns
    -------
    Tensor
        Concatenated tensor.
    """
    data = tensors[0].device.module.concatenate([t.data for t in tensors], axis=dim)
    return Tensor(data)


def diagonal(x: Tensor) -> Tensor:
    """Expands a tensor by turning the last dim into a diagonal matrix.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Expanded tensor.
    """
    return x.view((*x.shape, 1)) * identity(x.shape[-1])


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


def permute(x: Tensor, dims: tuple[int, ...]) -> Tensor:
    """Permutes a tensor's dimensions.

    Parameters
    ----------
    x: Tensor
        Input tensor.
    dims : tuple[int, ...]
        Rearranged dimensions.

    Returns
    -------
    Tensor
        Permuted tensor.
    """
    return x.permute(dims)


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

    # compute strided shape
    out = (x.shape[-1] - window_size) // stride + 1
    out_shape = (*x.shape[:-1], out, window_size)

    # compute strides
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

    # compute strided shape
    out = (x.shape[-1] - window_size) // stride + 1
    out_shape = (*x.shape[:-2], out, out, window_size, window_size)

    # compute strides
    x_str = x.strides
    out_strides = (*x_str[:-2], x_str[-2] * stride, x_str[-1] * stride, *x_str[-2:])

    str_func = x.device.module.lib.stride_tricks.as_strided
    return Tensor(str_func(x.data, out_shape, out_strides))


def repeat1d(x: Tensor, n: int) -> Tensor:
    """Repeat elements of a tensor along the last dimension.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    n : int
        Number of repeats.

    Returns
    -------
    Tensor
        Tensor with repeated values.
    """

    # compute strided shape
    out_shape = (*x.shape, n)

    # compute strides
    out_strides = (*x.strides, 0)

    str_func = x.device.module.lib.stride_tricks.as_strided
    y = str_func(x.data, out_shape, out_strides)
    return Tensor(y.reshape((*y.shape[:-2], y.shape[-2] * n)))


def repeat2d(x: Tensor, n: int) -> Tensor:
    """Repeat elements of a tensor along the last two dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    n : int
        Number of repeats.

    Returns
    -------
    Tensor
        Tensor with repeated values.
    """

    # compute strided shape
    out_shape = (*x.shape[:-1], n, x.shape[-1], n)

    # compute strides
    out_strides = (*x.strides[:-1], 0, x.strides[-1], 0)

    str_func = x.device.module.lib.stride_tricks.as_strided
    y = str_func(x.data, out_shape, out_strides)
    return Tensor(y.reshape((*y.shape[:-4], y.shape[-4] * n, y.shape[-2] * n)))


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


def split(x: Tensor, splits: int | Sequence[int], dim: int = -1) -> list[Tensor]:
    """Returns a list of new tensors by splitting the tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    splits : int | list[int]
        | Where to split the tensor.
        | ``int``: the tensor is split into n equally sized tensors.
        | ``Sequence[int]``: the tensor is split at the given indices.
    dim : int, optional
        Dimension along which to split the tensor. Defaults to ``-1``.

    Returns
    -------
    list[Tensor]
        List of tensors containing the split data.
    """
    return [Tensor(s) for s in x.device.module.split(x.data, splits, dim)]


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


def stack(tensors: Sequence[Tensor], dim: DimLike = 0) -> Tensor:
    """Returns a new tensor by stacking a sequence of tensors along a given dimension.

    Parameters
    ----------
    tensors : Sequence[Tensor]
        Sequence of Tensors to be stacked.
    dim : DimLike, optional
        Dimension along which to stack the tensors. Defaults to ``0``.

    Returns
    -------
    Tensor
        Stacked tensor.
    """
    return Tensor(tensors[0].device.module.stack([t.data for t in tensors], dim))


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


def transpose(x: Tensor, dim1: int, dim2: int) -> Tensor:
    """Transposes a tensor by swapping two dimensions.

    Parameters
    ----------
    x: Tensor
        Input tensor.
    dim1, dim2 : int
        Dimensions to transpose.

    Returns
    -------
    Tensor
        Transposed tensor.
    """
    return x.transpose(dim1, dim2)
