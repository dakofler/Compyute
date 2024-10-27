"""Dataloaders."""

import math
from collections.abc import Callable, Iterator
from functools import wraps
from typing import Any

from ...backend import Device, cpu
from ...random.random import permutation
from ...tensor_ops.creation_ops import arange
from ...tensor_ops.shape_ops import concat
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

    def __init__(
        self,
        data: tuple[Tensor, ...],
        batch_size: int = 1,
        device: Device = cpu,
        shuffle_data: bool = True,
        drop_remaining: bool = False,
    ) -> None:
        self.data = data
        self._n = len(self.data[0])
        self.batch_size = min(batch_size, self._n)
        self.device = device
        self.shuffle = shuffle_data
        self._additional_batch = not drop_remaining and self._n % self.batch_size > 0

    def __call__(self) -> Iterator[tuple[Tensor, ...]]:
        """Yields batched data.

        Yields
        -------
        Tensor
            Batched features.
        Tensor
            Batched labels.

        """
        idx = permutation(self._n) if self.shuffle else arange(self._n, dtype=int64)

        for i in range(len(self)):
            batch_idx = idx[i * self.batch_size : (i + 1) * self.batch_size]
            yield tuple(t[batch_idx].to_device(self.device) for t in self.data)

    def __len__(self) -> int:
        return max(1, self._n // self.batch_size + self._additional_batch)


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
        return concat(ys, dim=0)

    return wrapper
