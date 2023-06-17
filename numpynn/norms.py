"""Normalization functions"""

from numpynn.layers import Layer


class Normalization(Layer):
    """Normalization layer base class"""

    def __init__(self) -> None:
        super().__init__()
        self.is_activation_layer = True


class Layernorm(Normalization):
    """Implements layer normalization."""

    def compile(self, i, prev_layer, succ_layer) -> None:
        super().compile(i, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        self.y = self.x

    def backward(self) -> None:
        self.dx = self.dy
