"""Neural network parameter module"""

from walnut.tensor import Tensor
from walnut.cuda import numpy_to_cupy, cupy_to_numpy, ArrayLike, ScalarLike


__all__ = ["Parameter"]


class Parameter(Tensor):
    """Trainable neural network parameter."""

    def __init__(
        self,
        data: ArrayLike | ScalarLike,
        dtype: str = "float32",
        copy: bool = False,
        device: str = "cpu",
        label: str | None = None,
    ) -> None:
        """Trainable neural network parameter.

        Parameters
        ----------
        data : ArrayLike | ScalarLike
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
        self.optimizer_params = {}

    def to_device(self, device: str) -> None:
        """Moves the tensor to a specified device.

        Parameters
        ----------
        device : str
            Device to move the tensor to. Valid options are "cpu" and "cuda".

        Raises
        ----------
        AttributeError
            If device is not "cpu" or "cuda".

        """
        super().to_device(device)

        if device == "cpu":
            for key in self.optimizer_params:
                self.optimizer_params[key] = cupy_to_numpy(self.optimizer_params[key])
        else:
            for key in self.optimizer_params:
                self.optimizer_params[key] = numpy_to_cupy(self.optimizer_params[key])
