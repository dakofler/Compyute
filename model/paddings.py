import numpy as np


def Zero(image: np.ndarray):
    image_p = image.copy()
    width = image.shape[1]
    height = image.shape [0]

    zeros_x = np.zeros((width,)) 
    zeros_y = np.zeros((height + 2,))

    image_p = np.insert(image_p, 0, zeros_x, axis=0)
    image_p = np.insert(image_p, len(image_p), zeros_x, axis=0)
    image_p = np.insert(image_p, 0, zeros_y, axis=1)
    image_p = np.insert(image_p, len(image_p[0]), zeros_y, axis=1)

    return image_p

def Same(image: np.ndarray):
    return image.copy()