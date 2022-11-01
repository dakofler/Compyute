import numpy as np
import time


def Accuracy(x, y, model, loss_function):
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

        prediction[prediction == prediction.max()] = 1
        prediction[prediction != 1] = 0
        if not np.array_equal(prediction, y[i]):
            c = c + 1

    end = time.time()
    step = round((end - start) * 1000, 2)

    loss = np.sum(total_loss) / len(total_loss)
    accuracy = round(1 - 1 / n * c, 2)

    return loss, name, accuracy, step