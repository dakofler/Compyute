"""Neural network models module"""

from walnut.tensor import Tensor
import walnut.tensor_utils as tu


__all__ = ["DataLoader"]


class DataLoader:
    """DataLoader base class."""

    def __init__(self, X: Tensor, y: Tensor | None = None, batch_size: int = 1) -> None:
        """DataLoader base class.

        Parameters
        ----------
        X : Tensor
            Freature tensor.
        y : Tensor, optional
            Label tensor, by default None.
        batch_size : int, optional
            Size of returned batches, by default 1.
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __call__(self, shuffle: bool = True):
        """Returns batched data.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to shuffle the data each time the dataloader is called, by default True.

        Yields
        ------
        tuple[Tensor, Tensor]
            Batch of features and labels.
        """
        n = len(self.X)
        n_steps = len(self)
        b = min(self.batch_size, n)
        n_regular = n_steps * b

        if shuffle:
            indices = tu.random_permutation(n)
            self.X = self.X[indices]
            if self.y is not None:
                self.y = self.y[indices]

        # if labels are provided
        if self.y is not None:
            for i in range(n_steps):
                yield (self.X[i * b : (i + 1) * b], self.y[i * b : (i + 1) * b])
            # yield remaining samples, if there are any
            if n_regular < len(self.X):
                yield (self.X[n_regular:], self.y[n_regular:])
        else:
            for i in range(n_steps):
                yield (self.X[i * b : (i + 1) * b], None)
            # yield remaining samples, if there are any
            if n_regular < len(self.X):
                yield (self.X[n_regular:], None)

    def __len__(self) -> int:
        return max(1, len(self.X) // self.batch_size)
