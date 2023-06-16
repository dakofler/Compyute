# evaluation metrics module

from numpynn import networks
import numpy as np
import time


def Accuracy(x: np.ndarray, y: np.ndarray, model: networks.Sequential) -> float:
    "Computes the accuracy score of a prediction compared to target values."
    start = time.time()

    output, loss = model(x, y)
    preds = np.zeros_like(output)
    preds[np.arange(0, preds.shape[0]), np.argmax(output, axis=1)] = 1
    accuracy = np.sum(np.sum(preds == y, axis=1) == preds.shape[1]) / x.shape[0]

    end = time.time()
    step = round((end - start) * 1000, 2)

    print('loss %.4f | %10s %.4f | time %.2f ms' % (loss, Accuracy.__name__, accuracy, step))

    return accuracy
