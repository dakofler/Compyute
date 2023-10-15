"""Neural network models module"""

from walnut.tensor import Tensor
import walnut.tensor_utils as tu


__all__ = ["DataLoader"]


class DataLoader:
    """DataLoader base class."""

    def __init__(self, x: Tensor, y: Tensor | None = None, batch_size: int = 1) -> None:
        """DataLoader base class.

        Parameters
        ----------
        x : Tensor
            Freature tensor.
        y : Tensor, optional
            Label tensor, by default None.
        batch_size : int, optional
            Size of returned batches, by default 1.
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __call__(self, shuffle: bool = True, drop_remaining: bool = False):
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
        tuple[Tensor, Tensor]
            Batch of features and labels.
        """
        n = len(self.x)
        n_steps = len(self)
        b = min(self.batch_size, n)
        n_trunc = n_steps * b

        if shuffle:
            indices = tu.random_permutation(n)
            self.x = self.x[indices]
            if self.y is not None:
                self.y = self.y[indices]

        # yield batches
        for i in range(n_steps):
            x = self.x[i * b : (i + 1) * b]
            y = self.y[i * b : (i + 1) * b] if self.y is not None else None
            yield (x, y)

        # yield remaining samples, if there are any
        if not drop_remaining and n_trunc < n:
            x_remain = self.x[n_trunc:]
            y_remain = self.y[n_trunc:] if self.y is not None else None
            yield (x_remain, y_remain)

    def __len__(self) -> int:
        return max(1, len(self.x) // self.batch_size)
