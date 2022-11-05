import numpy as np


def Zero(image: np.ndarray, kernel_overhang: int):
    image_p = image.copy()
    width = image.shape[1]
    height = image.shape[0]
    depth = image.shape[2]
    zeros_x = np.zeros((kernel_overhang, width, depth)) 
    zeros_y = np.zeros((height + 2 * kernel_overhang, kernel_overhang, depth))
    image_p = np.concatenate((zeros_x, image_p), axis=0)
    image_p = np.concatenate((image_p, zeros_x), axis=0)
    image_p = np.concatenate((zeros_y, image_p), axis=1)
    image_p = np.concatenate((image_p, zeros_y), axis=1)
    return image_p

def Same(image: np.ndarray, kernel_overhang: int):
    return image.copy()