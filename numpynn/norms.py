"""Normalization functions"""

from numpynn.layers import Layer


class Normalization(Layer):

    def __init__(self) -> None:
        super().__init__()
        self.is_activation_layer = True

    def compile(self, id, prev_layer, succ_layer):
        super().compile(id, prev_layer, succ_layer)

    def forward(self) -> None:
        super().forward()
    
    def backward(self) -> None:
        super().backward()


class Layernorm(Normalization):
    def __init__(self) -> None:
        super.__init__()

    def compile(self, id, prev_layer, succ_layer) -> None:
        super().compile(id, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        self.y = self.x

    def backward(self) -> None:
        self.dx = self.dy