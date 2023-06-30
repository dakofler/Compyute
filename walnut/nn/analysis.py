"""Model analysis module"""

import numpy as np
import matplotlib.pyplot as plt

from walnut.tensor import Tensor, NumpyArray, ShapeLike
import walnut.tensor_utils as tu


def plot_curve(
    traces: dict[str, list[float]], figsize: ShapeLike, x_label: str, y_label: str
) -> None:
    """Plots one or multiple line graphs.

    Parameters
    ----------
    traces : dict[list[float]]
        Dictionary of labels and value lists to plot.
    figsize : ShapeLike
        Size of the plot.
    """
    plt.figure(figsize=figsize)
    legends = []
    for label in traces:
        values = traces[label]
        plt.plot(np.arange(1, len(values) + 1), values)
        legends.append(f"{label:s}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(legends, frameon=False)


def plot_distrbution(
    data: dict[str, NumpyArray],
    figsize: ShapeLike,
    title: str = "distribution",
    bins: int = 100,
) -> None:
    """Plots a line histogram.

    Parameters
    ----------
    data : dict[str, NumpyArray]
        Dictionary of labels and arrays.
    figsize : ShapeLike
        Size of the plot.
    bins : int, optional
        Number of bins used in the histogram, by default 100.
    """
    plt.figure(figsize=figsize)
    legends = []
    for label in data:
        values = data[label]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{label:10s} | mean {mean:.4f} | std {std:.4f}")
        y_vals, x_vals = np.histogram(values, bins=bins)
        x_vals = np.delete(x_vals, -1)
        plt.plot(x_vals, y_vals)
        legends.append(f"{label:s}")
        plt.legend(legends, frameon=False)
        plt.title(title)


def plot_images(
    data: dict[str, NumpyArray], figsize: ShapeLike, cmap: str = "viridis"
) -> None:
    """Plots array values as images.

    Parameters
    ----------
    data : dict[str, NumpyArray]
        Dictionary of array names and values.
    figsize : ShapeLike
        Size of the plot.
    cmap : str
        Colormap used in the plot.
    """
    for label in data:
        values = data[label]

        print(label)
        plt.figure(figsize=figsize)
        vmin = np.min(values).item()
        vmax = np.max(values).item()
        if values.ndim != 3:
            values = np.expand_dims(values, 0)
        for i in range(values.shape[0]):
            plt.subplot(10, 8, i + 1)
            plt.imshow(values[i, :, :], vmin=vmin, vmax=vmax, cmap=cmap)
            plt.xlabel(f"channel {str(i + 1)}")
        plt.show()


def plot_confusion_matrix(
    X: Tensor, Y: Tensor, figsize: ShapeLike, cmap: str = "Blues"
) -> None:
    """Plots the confusion matrix for predictions and targets.

    Parameters
    ----------
    X : Tensor
        A model's predictions.
    Y : Tensor
        Target values.
    figsize : ShapeLike
        Size of the plot.
    """

    # create tensor with ones where highest probabilities occur
    preds = tu.zeros_like(X).data
    max_prob_indices = np.argmax(X.data, axis=1)
    preds[np.arange(0, preds.shape[0]), max_prob_indices] = 1

    # get indeces of correct labels from Y
    y_index = np.argmax(Y.data, axis=1)

    classes = Y.shape[1]
    matrix = np.zeros((classes, classes))
    for i, pred in enumerate(preds):
        matrix[y_index[i]] += pred

    plt.figure(figsize=figsize)
    plt.tick_params(left=False, bottom=False, labelbottom=False, labeltop=True)
    plt.xticks(ticks=np.arange(0, classes, 1), labels=np.arange(0, classes, 1))
    plt.yticks(ticks=np.arange(0, classes, 1), labels=np.arange(0, classes, 1))
    plt.imshow(matrix, cmap=cmap)
    for (j, i), label in np.ndenumerate(matrix):
        plt.text(i, j, str(int(label)), ha="center", va="center")


def plot_probabilities(X: Tensor, figsize: ShapeLike) -> None:
    """Plots model predictions as a bar chart.

    Parameters
    ----------
    X : Tensor
        A model's predictions.
    figsize : ShapeLike
        Size of the plot.
    """
    classes = X.shape[1]
    plt.figure(figsize=figsize)
    plt.xticks(ticks=np.arange(0, classes, 1), labels=np.arange(0, classes, 1))
    plt.bar(np.arange(0, 10), X.reshape((10,)).data)
    plt.xlabel("class")
    plt.ylabel("probability")
