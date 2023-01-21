import numpy as np
import time
from model.networks import Network


def Accuracy(x: np.ndarray, y: np.ndarray, model: Network, loss_function) -> (float|str|float|float):
    "Computes the accuracy score of a prediction compared to target values."
    name = 'accuracy'
    n = len(x)
    c = 0
    total_loss = np.zeros((x.shape[0]))
    start = time.time()

    for i, p in enumerate(x):
        print(f'Evaluating ... {i + 1}/{n}', end='\r')
        prediction = model.predict(p)
        loss, loss_gradient = loss_function(prediction, np.squeeze(y[i]))
        total_loss[i] = loss

        prediction = np.round(prediction, 0)
        if not np.array_equal(prediction, y[i]):
            c = c + 1

    end = time.time()
    step = round((end - start) * 1000, 2)

    loss = np.sum(total_loss) / len(total_loss)
    accuracy = round(1 - 1 / n * c, 2)

    return loss, name, accuracy, step