"""Neural network parameter module"""

from walnut.tensor import Tensor, ArrayLike, NpTypeLike, PyTypeLike


__all__ = ["Parameter"]


class Parameter(Tensor):
    """Trainable neural network parameter."""

    def __init__(
        self,
        data: ArrayLike | NpTypeLike | PyTypeLike,
        dtype: str = "float32",
        copy: bool = False,
        device: str = "cpu",
        label: str | None = None,
    ) -> None:
        """Trainable neural network parameter.

        Parameters
        ----------
        data : NpArrayLike | NpTypeLike | PyTypeLike
            Data to initialize the tensor.
        dtype: str, optional
            Datatype of the tensor data, by default "float32".
        copy: bool, optional
            If true, the data object is copied (may impact performance), by default False.
        device: str, optinal
            The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".
        label: str | None, optional
            Parameter label, by default None.
        """
        if isinstance(data, Tensor):
            data = data.data
        super().__init__(data, dtype, copy, device)
        self.label = label
        self.temp_params = {}
