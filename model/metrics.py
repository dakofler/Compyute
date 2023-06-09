import numpy as np
import time
from model.networks import Network


def Accuracy(x: np.ndarray, y: np.ndarray, model: Network, loss_function) -> None:
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
    accuracy = 1 - 1 / n * c
    print('loss %.4f | %10s %.4f | time %.2f ms' % (loss, name, accuracy, step))