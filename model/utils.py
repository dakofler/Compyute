import pandas as pd
import numpy as np
from numpy.fft  import fft2, ifft2
from scipy import signal


def split_train_test_data(array, ratio=0.3):
    shuffle_index = np.arange(len(array))
    np.random.shuffle(shuffle_index)
    array_shuffled = array[shuffle_index]
    i = int(len(array_shuffled) * (1 - ratio))
    train_array = array_shuffled[:i]
    test_array = array_shuffled[i:]
    return train_array, test_array

def expand_dims(array: np.ndarray, dims):
    while array.ndim < dims:
        array = np.expand_dims(array, -1)
    return array

def split_X_Y(data, num_x_cols):
    X = data[:, :num_x_cols]
    Y = data[:, num_x_cols:]
    return X, Y

def categorical_to_numeric(data: pd.DataFrame):
    return pd.get_dummies(data)

def normalize(array: np.ndarray, axis = 0):
    return array / array.max(axis=axis)

def shuffle(x: np.ndarray, y: np.ndarray):
    shuffle_index = np.arange(len(x))
    np.random.shuffle(shuffle_index)
    x_shuffled = x[shuffle_index]
    y_shuffled = y[shuffle_index]
    return x_shuffled, y_shuffled

def heatmap(array):
    im = np.zeros(array.shape[:2])
    for i in range(array.shape[2]):
        im += normalize(array[:, :, i], axis=None)
    return normalize(im, axis=None)

def convolve(image: np.ndarray, f: np.ndarray, s = 1):
    o_y = int((image.shape[0] - f.shape[0]) / s) + 1
    f_y = f.shape[0]
    f_x = f.shape[1]
    o = np.zeros((o_y, o_y))
    f = np.fliplr(f)

    y_count = 0
    for y in range(0, o_y * s, s):
        x_count = 0
        for x in range(0, o_y * s, s):
            array = image[y : y + f_y, x : x + f_x]
            o[y_count, x_count] = np.sum(array * f)
            x_count += 1
        y_count += 1
    return o

def convolve_scipy(image: np.ndarray, f: np.ndarray):
    return signal.convolve2d(image, f, mode='valid')

def convolve_tensor_fft(tensor: np.ndarray, kernel: np.ndarray): # tensor shape (h, w, c), kernel shape (k, c, h, w)
    
    C = np.zeros((*tensor.shape[:2], kernel.shape[0]))
    kernel_fft = np.moveaxis(fft2(kernel, s=tensor.shape[:2]), 1, -1)
    image_fft = fft2(tensor, axes=(0, 1))
    for k in np.arange(kernel.shape[0]):
        C[:, :, k] = np.sum(np.real(ifft2(image_fft * kernel_fft[k], axes=(0, 1))), axis=2)
    p = int(kernel.shape[2] / 2)
    return C[p : -p, p : -p]

def convolve_2d_fft(array: np.ndarray, kernel: np.ndarray):
    kernel_fft = fft2(kernel, s=array.shape)
    image_fft = fft2(array)
    return np.real(ifft2(image_fft * kernel_fft))

# plot-util for multiple plots in a grid? https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html