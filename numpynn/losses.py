# loss functions module

import numpy as np


class Loss():
    
    def __init__(self) -> None:
        self.loss = 0
        self._y = None
        self._t = None
    
    def set_vals(self, y, t) -> None:
        self._y = y + 1e-7 # to avoid dividing by 0
        self._t = t + 1e-7 # to avoid dividing by 0


class MSE(Loss):

    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, y, t) -> np.ndarray:
        super().set_vals(y, t)
        return 0.5 * np.sum((self._t - self._y)**2)
    
    def backward(self) -> np.ndarray:
        return self._y - self._t


class Crossentropy(Loss):

    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, y, t) -> np.ndarray:
        super().set_vals(y, t)
        return -np.mean(np.log(self._y) * self._t)
    
    def backward(self) -> np.ndarray:
        return -self._t / (2 * self._y * self._t.shape[0])
