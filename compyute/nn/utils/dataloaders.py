"""Dataloaders."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from functools import wraps

from ...backend import Device, cpu
from ...random.random import permutation
from ...tensor_ops.creating import arange, concat
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
    _iterator = 0
    _idx: Tensor
    _n_samples: int
    _n_steps: int

    def __init__(
        self,
        data: tuple[Tensor, ...],
        batch_size: int = 1,
        device: Device = cpu,
        shuffle_data: bool = True,
        drop_remaining: bool = False,
    ) -> None:
        self.data = data
        self._n_samples = len(self.data[0])
        self.batch_size = min(batch_size, self._n_samples)
        self.device = device
        self.shuffle = shuffle_data
        self._n_steps = self._n_samples // self.batch_size
        if drop_remaining and self._n_steps * self.batch_size < self._n_samples:
            self._n_steps += 1

    def __iter__(self) -> Dataloader:
        self._iterator = 0
        if self.shuffle:
            self._idx = permutation(self._n_samples)
        else:
            self._idx = arange(self._n_samples, dtype=int64)
        return self

    def __next__(self) -> tuple[Tensor, ...]:
        if self._iterator < self._n_steps:
            max_idx = min((self._iterator + 1) * self.batch_size, self._n_samples)
            batch_idx = self._idx[self._iterator * self.batch_size : max_idx + 1]
            self._iterator += 1
            return tuple(t[batch_idx].to_device(self.device) for t in self.data)
        raise StopIteration

    def __len__(self) -> int:
        return self._n_steps


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
    def wrapper(x: Tensor, *args, **kwargs) -> Tensor:
        dataloader = Dataloader((x,), batch_size, device, shuffle_data, drop_remaining)
        ys = [func(*x_batch, *args, **kwargs) for x_batch in dataloader]
        return concat(ys, axis=0)

    return wrapper
