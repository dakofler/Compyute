"""Dataloaders."""

import math
from collections.abc import Callable, Iterator
from functools import wraps
from typing import Any

from ...backend import Device, cpu
from ...random.random import permutation
from ...tensor_ops.creation_ops import arange, concat
from ...tensors import Tensor
from ...typing import int64

__all__ = ["Dataloader", "batched"]


class Dataloader:
    """DataLoader to yield batched data for training and inference.

    Parameters
    ----------
    data : tuple[Tensor, ...]
        Data to load.
    batch_size : int, optional
        Size of returned batches. Defaults to ``1``.
    device : Device, optional
        Device the tensors should be loaded to. Defaults to :class:`compyute.cpu`.
    shuffle_data : bool, optional
        Whether to shuffle the data each time the dataloader is called. Defaults to ``True``.
    drop_remaining : bool, optional
        Whether to drop data, that remains when the number of samples is not divisible by
        ``batch_size``. Defaults to ``False``.
    """

    data: tuple[Tensor, ...]
    batch_size: int
    device: Device
    shuffle: bool
    drop_remaining: bool

    def __init__(
        self,
        data: tuple[Tensor, ...],
        batch_size: int = 1,
        device: Device = cpu,
        shuffle_data: bool = True,
        drop_remaining: bool = False,
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle_data
        self.drop_remaining = drop_remaining

    def __call__(self) -> Iterator[tuple[Tensor, ...]]:
        """Yields batched data.

        Yields
        -------
        Tensor
            Batched features.
        Tensor
            Batched labels.

        """
        t1 = self.data[0]
        n = t1.shape[0]
        n_steps = len(self)
        b = min(self.batch_size, n)

        idx = permutation(n) if self.shuffle else arange(n, dtype=int64)

        for i in range(n_steps):
            batch_idx = idx[i * b : (i + 1) * b]
            yield tuple(t[batch_idx].to_device(self.device) for t in self.data)

        if not self.drop_remaining and n_steps * b < n:
            n_trunc = n_steps * b
            yield tuple(t[idx[n_trunc:]].to_device(self.device) for t in self.data)

    def __len__(self) -> int:
        return max(1, math.ceil(self.data[0].shape[0] / self.batch_size))


def batched(
    func: Callable[[Tensor], Tensor],
    batch_size: int = 1,
    device: Device = cpu,
    shuffle_data: bool = True,
    drop_remaining: bool = False,
) -> Callable:
    """Decorator for performing batched inference.

    Parameters
    ----------
    batch_size : int, optional
        Size of returned batches. Defaults to ``1``.
    device : Device, optional
        Device the tensors should be loaded to. Defaults to :class:`compyute.cpu`.
    shuffle_data : bool, optional
        Whether to shuffle the data each time the dataloader is called. Defaults to ``True``.
    drop_remaining : bool, optional
        Whether to drop data, that remains when the number of samples is not divisible by
        the ``batch_size``.
    """

    @wraps(func)
    def wrapper(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        dataloader = Dataloader((x,), batch_size, device, shuffle_data, drop_remaining)
        ys = [func(*x_batch, *args, **kwargs) for x_batch in dataloader()]
        return concat(ys, axis=0)

    return wrapper
