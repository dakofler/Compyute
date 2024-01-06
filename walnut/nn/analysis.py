"""Model analysis module"""

import numpy as np
import matplotlib.pyplot as plt

from walnut.tensor import Tensor, ArrayLike, ShapeLike
import walnut.tensor_utils as tu
from walnut.nn.funcional import softmax
from walnut.nn.models import Model

__all__ = [
    "plot_curve",
    "plot_distrbution",
    "plot_images",
    "plot_confusion_matrix",
    "plot_probabilities",
]


def plot_curve(
    traces: dict[str, list[float]],
    figsize: ShapeLike,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Plots one or multiple line graphs.

    Parameters
    ----------
    traces : dict[list[float]]
        Dictionary of labels and value lists to plot.
    figsize : ShapeLike
        Size of the plot.
    title : str
        Plot title.
    x_label : str
        Label for the x axis.
    y_label : str
        Label for the x axis.
    """
    plt.figure(figsize=figsize)
    legends = []
    for label in traces:
        values = traces[label]
        plt.plot(np.arange(1, len(values) + 1), values)
        legends.append(f"{label:s}")
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(legends)


def plot_distrbution(
    data: dict[str, ArrayLike],
    figsize: ShapeLike,
    title: str = "distribution",
    bins: int = 100,
) -> None:
    """Plots a line histogram.

    Parameters
    ----------
    data : dict[str, NpArrayLike]
        Dictionary of labels and arrays.
    figsize : ShapeLike
        Size of the plot.
    title : str, optional
        Plot title, by default "distribution".
    bins : int, optional
        Number of bins used in the histogram, by default 100.
    """
    plt.figure(figsize=figsize)
    legends = []
    for label in data:
        values = data[label]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{label:10s} | mean {mean:9.4f} | std {std:9.4f}")
        y_vals, x_vals = np.histogram(values, bins=bins)
        x_vals = np.delete(x_vals, -1)
        plt.plot(x_vals, y_vals)
        legends.append(f"{label:s}")
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.title(title)
    plt.xlabel("values")
    plt.ylabel("count")
    plt.legend(legends)


def plot_images(
    data: dict[str, ArrayLike],
    figsize: ShapeLike,
    cmap: str = "gray",
    plot_axis: bool = False,
) -> None:
    """Plots array values as images.

    Parameters
    ----------
    data : dict[str, NpArrayLike]
        Dictionary of array names and values.
    figsize : ShapeLike
        Size of the plot.
    cmap : str, optional
        Colormap used in the plot, by default "gray".
    plot_axis : bool, optional
        Whether to plot axes, by default False.
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
            if not plot_axis:
                plt.tick_params(
                    left=False, bottom=False, labelleft=False, labelbottom=False
                )
        plt.show()


def plot_confusion_matrix(
    x: Tensor, y: Tensor, figsize: ShapeLike, cmap: str = "Blues"
) -> None:
    """Plots the confusion matrix for predictions and targets.

    Parameters
    ----------
    x : Tensor
        A model's predictions.
    y : Tensor
        Target values.
    figsize : ShapeLike
        Size of the plot.
    cmap : str, optional
        Colormap used for the plot, by default "Blues".
    """

    classes = int(y.max().item() + 1)
    preds = x.argmax(-1)
    matrix = tu.zeros((classes, classes))
    for i in range(preds.len):
        matrix[y[i].item(), preds[i].item()] += 1

    plt.figure(figsize=figsize)
    plt.tick_params(left=False, bottom=False, labelbottom=False, labeltop=True)
    plt.xticks(ticks=np.arange(0, classes, 1), labels=np.arange(0, classes, 1))
    plt.yticks(ticks=np.arange(0, classes, 1), labels=np.arange(0, classes, 1))
    plt.imshow(matrix.data, cmap=cmap)
    for (j, i), label in np.ndenumerate(matrix.data):
        plt.text(i, j, str(int(label)), ha="center", va="center")


def plot_probabilities(x: Tensor, figsize: ShapeLike) -> None:
    """Plots model predictions as a bar chart.

    Parameters
    ----------
    x : Tensor
        A model's logits.
    figsize : ShapeLike
        Size of the plot.
    """
    classes = x.shape[1]
    preds = softmax(x)
    plt.figure(figsize=figsize)
    plt.xticks(ticks=np.arange(0, classes, 1), labels=np.arange(0, classes, 1))
    plt.bar(np.arange(0, classes), preds.reshape((classes,)).data)
    plt.xlabel("class")
    plt.ylabel("probability")


def model_summary(model: Model, input_shape: ShapeLike) -> None:
    """Prints information about a model.

    Parameters
    ----------
    model : Model
        Neural network model.
    input_shape : ShapeLike
        Shape of the model input ignoring the batch dimension.
    """
    n = 63
    summary = ["-" * n]
    summary.append(f"\n{'Layer':25s} {'Output Shape':20s} {'# Parameters':>15s}\n")
    summary.append("=" * n)
    summary.append("\n")

    x = tu.ones((1,) + input_shape)
    x.to_device(model.device)
    model.remember = True
    _ = model(x)

    def build_string(module, summary, depth):
        name = " " * depth + module.__class__.__name__
        output_shape = str((-1,) + module.y.shape[1:])
        n_params = sum(p.data.size for p in module.parameters())
        summary.append(f"{name:25s} {output_shape:20s} {n_params:15d}\n")

        for sub_module in module.sub_modules:
            build_string(sub_module, summary, depth + 1)

    build_string(model, summary, 0)
    summary.append("=" * n)
    tot_parameters = sum(p.data.size for p in model.parameters())

    model.reset()
    model.remember = False
    string = "".join(summary)
    print(f"{string}\n\nTotal parameters: {tot_parameters}")
