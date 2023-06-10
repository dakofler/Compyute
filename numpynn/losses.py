import numpy as np


class Loss():
    def __init__(self) -> None:
        self.loss = 0
        self._y = None
        self._t = None
    
    def set_vals(self, y, t):
        self._y = y
        self._t = t

class MSE(Loss):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, y, t):
        super().set_vals(y, t)
        return 0.5 * np.sum((self._t - self._y)**2)
    
    def backward(self):
        return self._y - self._t

class crossentropy(Loss):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, y, t):
        super().set_vals(y, t)
        # logprobs = np.log(self._y)
        # loss = -np.mean(logprobs * self._t)
        # return loss
        return -np.mean(np.log(self._y) * self._t)
    
    def backward(self):
        # dlogprobs = -self._t / self._t.shape[0]
        # dprobs = 1.0 / self._y * dlogprobs
        # return dprobs
        return -self._t / (2 * self._y * self._t.shape[0])