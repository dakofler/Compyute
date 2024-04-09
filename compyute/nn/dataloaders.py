"""Dataloaders module"""

from typing import Generator, Optional
from ..random import permutation
from ..tensor import Tensor


__all__ = ["DataLoader"]


class DataLoader:
    """DataLoader base class."""

    def __init__(
        self, x: Tensor, y: Optional[Tensor] = None, batch_size: int = 1
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
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __call__(
        self, shuffle: bool = True, drop_remaining: bool = False
    ) -> Generator[Tensor, None, None] | Generator[tuple[Tensor, Tensor], None, None]:
        """Returns data in a batched form.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to shuffle the data each time the dataloader is called, by default True.
        drop_remaining: bool, optional
            Whether to drop data, that remains when the number of samples is not divisible by
            the batch_size.

        Yields
        ------
        Generator[Tensor, None, None] | Generator[tuple[Tensor, Tensor], None, None]
            Batch of inputs or batch of inputs and targets,
            depending on whether targets were provided.
        """
        n = self.x.shape[0]
        n_steps = len(self)
        b = min(self.batch_size, n)
        n_trunc = n_steps * b
        shuffle_idx = permutation(n, device=self.x.device)

        # if y is None
        x = self.x if not shuffle else self.x[shuffle_idx]

        if self.y is None:
            for i in range(n_steps):
                yield x[i * b : (i + 1) * b]

            if not drop_remaining and n_trunc < n:
                yield x[n_trunc:]
        else:
            y = self.y if not shuffle else self.y[shuffle_idx]

            for i in range(n_steps):
                yield x[i * b : (i + 1) * b], y[i * b : (i + 1) * b]

            if not drop_remaining and n_trunc < n:
                yield x[n_trunc:], y[n_trunc:]

    def __len__(self) -> int:
        return max(1, self.x.shape[0] // self.batch_size)
