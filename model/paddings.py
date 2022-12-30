import numpy as np
import math

def Valid(image: np.ndarray, kernel_size = None):
    return image.copy()

def Same(image: np.ndarray, kernel_size = None):
    width = math.floor(kernel_size[0] / 2)
    return np.pad(image, width)[:, :, width : -width]