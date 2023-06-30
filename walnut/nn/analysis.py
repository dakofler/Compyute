"""Model analysis module"""

import numpy as np
import matplotlib.pyplot as plt

from walnut.tensor import NumpyArray, ShapeLike


def plot_curve(traces: dict[str, list[float]], figsize: ShapeLike) -> None:
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
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(legends)


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
        plt.legend(legends)
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
