"""Normalization functions"""

from numpynn import layers
import numpy as np

class Batchnorm(layers):
    def __init__(self) -> None:
        super.__init__()

    def compile(self, id: int, prev_layer: object, succ_layer: object) -> None:
        super().compile(id, prev_layer, succ_layer)
        self.forward()

    def forward(self) -> None:
        self.y = self.x

    def backward(self) -> None:
        self.dx = self.dy