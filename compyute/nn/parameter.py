"""Neural network parameter module"""

from ..tensor import Tensor


__all__ = ["Parameter"]


class Parameter(Tensor):
    """Trainable neural network parameter."""

    def __init__(
        self,
        data: Tensor,
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
        device: str, optional
            The device the tensor is stored on ("cuda" or "cpu"), by default "cpu".
        label: str | None, optional
            Parameter label, by default None.
        """
        super().__init__(data.data, dtype, copy, device)
        self.label = label
        self.optimizer_params: dict[str, Tensor] = {}

    def __repr__(self) -> str:
        return f"Parameter {self.label}:\n{super().__repr__()}"

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
        for param in self.optimizer_params.values():
            param.to_device(device)
