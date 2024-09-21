"""Neural network embedding modules."""

from typing import Optional

from ...tensor_ops.creation_ops import empty
from ...tensors import Tensor
from ...typing import DType
from ..functional.embedding_funcs import EmbeddingFn
from ..parameter import Parameter, update_parameter_grad
from ..utils.initializers import init_normal
from .module import Module

__all__ = ["Embedding"]


class Embedding(Module):
    r"""Lookup embedding layer.

    Shapes:
        - Input :math:`(B_1, ... , B_n, S)`
        - Output :math:`(B_1, ... , B_n, S, E)`
    where
        - :math:`B_1, ... , B_n` ... batch axes
        - :math:`S` ... sequence
        - :math:`E` ... embedding dimension

    Parameters
    ----------
    n_embeddings : int
        Number of embedding vectors.
    embedding_dim : int
        Embedding vector dimensions.
    dtype : DType, optional
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Embeddings are initialized from :math:`\mathcal{N}(0, 1)`.
    """

    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim

        # init parameters
        self.w = Parameter(empty((n_embeddings, embedding_dim), dtype=dtype))
        self._init_parameters_and_buffers()

    def _init_parameters_and_buffers(self) -> None:
        init_normal(self.w)

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return EmbeddingFn.forward(self.fcache, x, self.w)

    @Module.register_backward
    def backward(self, dy: Tensor) -> None:
        dw = EmbeddingFn.backward(self.fcache, dy)
        update_parameter_grad(self.w, dw)
