# evaluation metrics module

from numpynn.networks import Sequential
from numpynn import inits
import numpy as np
import time


def Accuracy(x: np.ndarray, y: np.ndarray, model: Sequential) -> float:
    "Computes the accuracy score of a prediction compared to target values."
    start = time.time()
    output, loss = model(x, y)

    # create array with ones where highest probabilities occur
    preds = inits.zeros_like(output)
    pb, _ = preds.shape
    max_prob_indices = np.argmax(output, axis=1)
    preds[np.arange(0, pb), max_prob_indices] = 1

    # count number of correct samples
    num_correct_preds = np.sum(preds * y)
    accuracy = num_correct_preds / pb

    end = time.time()
    step = round((end - start) * 1000.0, 2)
    print('loss %.4f | %10s %.4f | time %.2f ms' % (loss, Accuracy.__name__, accuracy, step))
    return accuracy
