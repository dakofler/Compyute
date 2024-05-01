"""Dataloaders module"""

from typing import Callable, Generator, Optional
from ..random import shuffle
from ..tensor import Tensor
from ..tensor_f import concatenate
from ..types import DeviceLike


__all__ = ["DataLoader"]


class DataLoader:
    """DataLoader base class."""

    def __init__(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        batch_size: int = 1,
        device: DeviceLike = "cpu",
        shuffle_data: bool = True,
        drop_remaining: bool = False,
    ) -> None:
        """DataLoader base class.

        Parameters
        ----------
        x : Tensor
            Input tensor.
        y : Tensor, optional
            Target tensor, by default None.
        batch_size : int, optional
            Size of returned batches, by default 1.
        device: DeviceLike, optional
            Device the tensors should be loaded to ("cuda" or "cpu"), by default None.
        shuffle_data : bool, optional
            Whether to shuffle the data each time the dataloader is called, by default True.
        drop_remaining: bool, optional
            Whether to drop data, that remains when the number of samples is not divisible by
            the batch_size.
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.device = device
        self.drop_remaining = drop_remaining
        self.shuffle = shuffle_data

    def __call__(
        self,
    ) -> Generator[tuple[Tensor, Optional[Tensor]], None, None]:
        """Batch of inputs or batch of inputs and targets."""
        n = self.x.shape[0]
        n_steps = len(self)
        b = min(self.batch_size, n)
        n_trunc = n_steps * b

        # shuffle data
        if self.shuffle:
            self.x, idx = shuffle(self.x)
            if self.y is not None:
                self.y = self.y[idx]

        if self.y is None:
            for i in range(n_steps):
                x_batch = self.x[i * b : (i + 1) * b]
                x_batch.to_device(self.device)
                yield x_batch, None

            if not self.drop_remaining and n_trunc < n:
                x_batch = self.x[n_trunc:]
                x_batch.to_device(self.device)
                yield x_batch, None

        else:
            for i in range(n_steps):
                x_batch = self.x[i * b : (i + 1) * b]
                y_batch = self.y[i * b : (i + 1) * b]
                x_batch.to_device(self.device)
                y_batch.to_device(self.device)
                yield x_batch, y_batch

            if not self.drop_remaining and n_trunc < n:
                x_batch = self.x[n_trunc:]
                y_batch = self.y[n_trunc:]
                x_batch.to_device(self.device)
                y_batch.to_device(self.device)
                yield x_batch, y_batch

    def __len__(self) -> int:
        return max(1, self.x.shape[0] // self.batch_size)


def batched(
    outer_func: Callable,
    batch_size: int = 1,
    device: DeviceLike = "cpu",
    shuffle_data: bool = True,
    drop_remaining: bool = False,
) -> Callable:
    """Decorator for performing input batching.

    Parameters
    ----------
    outer_func : Callable
        Function to be called.
    batch_size : int, optional
        Size of returned batches, by default 1.
    device: DeviceLike, optional
        Device the tensors should be loaded to ("cuda" or "cpu"), by default None.
    shuffle_data : bool, optional
        Whether to shuffle the data each time the dataloader is called, by default True.
    drop_remaining: bool, optional
        Whether to drop data, that remains when the number of samples is not divisible by
        the batch_size.
    """

    def inner_func(x: Tensor):
        """Calls a function using batched inputs.

        Parameters
        ----------
        x : Tensor
            Input tensor.
        """
        dataloader = DataLoader(
            x,
            batch_size=batch_size,
            device=device,
            shuffle_data=shuffle_data,
            drop_remaining=drop_remaining,
        )
        outputs = []

        for batch, _ in dataloader():
            outputs.append(outer_func(batch))

        return concatenate(outputs, axis=0)

    return inner_func
