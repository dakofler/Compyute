import numpy as np


def Zero(image: np.ndarray, width: int = 1, axis: tuple = (0, 1)):
    return np.pad(image, width)[:, :, width : -width]

def Same(image: np.ndarray, width: int = 0, axis=(0, 1)):
    return image.copy()