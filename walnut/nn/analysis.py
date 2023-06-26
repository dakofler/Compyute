"""Model analysis module"""

from typing import Any
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def plot_curve(values: list[float]) -> None:
    """Plots a line graph based on a list of values.

    Parameters
    ----------
    values : list[float]
        List of values.
    """
    plt.figure(figsize=(20, 4))
    plt.plot(np.arange(len(values)), values)
    plt.xlabel("epoch")
    plt.ylabel("loss")


def plot_distrbution(
    data: dict[str, npt.NDArray[Any]], title: str = "distribution", bins: int = 100
) -> None:
    """Plots a line histogram.

    Parameters
    ----------
    data : dict[str, npt.NDArray[Any]]
        Dictionary of labels and arrays.
    bins : int, optional
        Number of bins used in the histogram, by default 100.
    """
    plt.figure(figsize=(20, 4))
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


def plot_images(data: dict[str, npt.NDArray[Any]]) -> None:
    """Plots array values as images.

    Parameters
    ----------
    data : dict[str, npt.NDArray[Any]]
        Dictionary of array names and values.
    """
    for label in data:
        values = data[label]
        print(label)
        plt.figure(figsize=(40, 40))
        vmin = np.min(values).item()
        vmax = np.max(values).item()
        for i in range(values.shape[1]):
            plt.subplot(10, 8, i + 1)
            plt.imshow(values[0, i, :, :], vmin=vmin, vmax=vmax)
            plt.xlabel(f"channel {str(i + 1)}")
        plt.show()
