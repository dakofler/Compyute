"""Neural network parameter module"""

from typing import Optional
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
        device: Optional[DeviceLike] = None,
        label: Optional[str] = None,
    ) -> None:
        """Trainable neural network parameter.

        Parameters
        ----------
        data : ArrayLike | ScalarLike
            Data to initialize the tensor.
        dtype: DtypeLike, optional
            Datatype of the tensor data, by default None. If None, the dtype is inferred.
        copy: bool, optional
            If true, the data object is copied (may impact performance), by default False.
        device: DeviceLike, optional
            Device the tensor is stored on ("cuda" or "cpu"), by default None.
        label: str, optional
            Parameter label, by default None.
        """
        super().__init__(data.data, dtype, copy, device)
        self.label = label
