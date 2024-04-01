"""Neural network parameter module"""

from ..tensor import Tensor
from ..types import DeviceLike, DtypeLike


__all__ = ["Parameter"]


class Parameter(Tensor):
    """Trainable neural network parameter."""

    def __init__(
        self,
        data: Tensor,
        dtype: DtypeLike = "float32",
        copy: bool = False,
        device: DeviceLike | None = None,
        label: str | None = None,
    ) -> None:
        """Trainable neural network parameter.

        Parameters
        ----------
        data : ArrayLike | ScalarLike
            Data to initialize the tensor.
        dtype: DtypeLike | None, optional
            Datatype of the tensor data, by default None. If None, the dtype is inferred.
        copy: bool, optional
            If true, the data object is copied (may impact performance), by default False.
        device: DeviceLike | None, optional
            Device the tensor is stored on ("cuda" or "cpu"), by default None.
        label: str | None, optional
            Parameter label, by default None.
        """
        super().__init__(data.data, dtype, copy, device)
        self.label = label
