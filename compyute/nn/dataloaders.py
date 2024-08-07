"""Dataloaders."""

from functools import wraps
from typing import Callable, Iterator, Optional

from ..base_tensor import Tensor
from ..engine import Device, _DeviceLike
from ..random.random import shuffle
from ..tensor_functions.creating import concatenate

__all__ = ["Dataloader"]


class Dataloader:
    """DataLoader to yield batched data for training and inference.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    y : Tensor, optional
        Target tensor. Defaults to ``None``.
    batch_size : int, optional
        Size of returned batches. Defaults to ``1``.
    device : DeviceLike, optional
        Device the tensors should be loaded to. Defaults to :class:`compyute.cpu`.
    shuffle_data : bool, optional
        Whether to shuffle the data each time the dataloader is called. Defaults to ``True``.
    drop_remaining : bool, optional
        Whether to drop data, that remains when the number of samples is not divisible by
        ``batch_size``. Defaults to ``False``.
    """

    def __init__(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        batch_size: int = 1,
        device: _DeviceLike = Device.CPU,
        shuffle_data: bool = True,
        drop_remaining: bool = False,
    ) -> None:
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.device = Device(device)
        self.drop_remaining = drop_remaining
        self.shuffle = shuffle_data

    def __call__(self) -> Iterator[tuple[Tensor, Optional[Tensor]]]:
        """Yields batched data.

        Yields
        -------
        Tensor
            Batched features.
        Tensor
            Batched labels.

        """
        n = self.x.shape[0]
        n_steps = len(self)
        b = min(self.batch_size, n)
        n_trunc = n_steps * b

        # shuffle data
        if self.shuffle:
            self.x, idx = shuffle(self.x)
            self.y = self.y[idx] if self.y is not None else None

        # yield batches
        for i in range(n_steps):
            x_batch = self.x[i * b : (i + 1) * b].to_device(self.device)

            if self.y is not None:
                y_batch = self.y[i * b : (i + 1) * b].to_device(self.device)
                yield x_batch, y_batch
            else:
                yield x_batch, None

        # yield remaining
        if not self.drop_remaining and n_trunc < n:
            x_batch = self.x[n_trunc:].to_device(self.device)

            if self.y is not None:
                y_batch = self.y[n_trunc:].to_device(self.device)
                yield x_batch, y_batch
            else:
                yield x_batch, None

    def __len__(self) -> int:
        return max(1, self.x.shape[0] // self.batch_size)


def batched(
    func: Callable[[Tensor], Tensor],
    batch_size: int = 1,
    device: _DeviceLike = Device.CPU,
    shuffle_data: bool = True,
    drop_remaining: bool = False,
) -> Callable:
    """Decorator for performing input batching.

    Parameters
    ----------
    batch_size : int, optional
        Size of returned batches. Defaults to ``1.
    device : DeviceLike, optional
        Device the tensors should be loaded to. Defaults to ``Device.CPU.
    shuffle_data : bool, optional
        Whether to shuffle the data each time the dataloader is called. Defaults to ``True.
    drop_remaining : bool, optional
        Whether to drop data, that remains when the number of samples is not divisible by
        the batch_size.
    """

    @wraps(func)
    def wrapper(x: Tensor, *args, **kwargs) -> Tensor:
        dataloader = Dataloader(
            x,
            batch_size=batch_size,
            device=device,
            shuffle_data=shuffle_data,
            drop_remaining=drop_remaining,
        )
        ys = [func(x_batch, *args, **kwargs) for x_batch, _ in dataloader()]
        return concatenate(ys, axis=0)

    return wrapper
