import numpy as np
import time
from numpynn.networks import Network


def Accuracy(x: np.ndarray, y: np.ndarray, model: Network) -> None:
    "Computes the accuracy score of a prediction compared to target values."
    start = time.time()
    c = 0

    print(f'Evaluating ...', end='\r')
    output = model.predict(x)
    loss = model.loss_function(output, y) # compute loss
    output = np.round(output, 0)
    if not np.array_equal(output, y):
        c += 1

    end = time.time()
    step = round((end - start) * 1000, 2)
    accuracy = 1 - 1 / x.shape[0] * c

    print('loss %.4f | %10s %.4f | time %.2f ms' % (loss, Accuracy.__name__, accuracy, step))