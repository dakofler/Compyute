import numpy as np


def root_mean_square(output: np.ndarray, target: np.ndarray) -> (float|np.ndarray):
    "Computes the root mean square error."
    loss = 0.5 * np.sum(np.power(target - output, 2))
    loss_gradient = output - target
    return loss, loss_gradient

def categorical_crossentropy(output: np.ndarray, target: np.ndarray) -> (float|np.ndarray):
    "Computes the categorical crossentropy error."
    output = output + 1e-7
    target = target + 1e-7
    loss = - np.sum(target * np.log(output))
    loss_gradient = output - target
    return loss, loss_gradient

def binary_crossentropy(output: np.ndarray, target: np.ndarray) -> (float|np.ndarray): 
    "Computes the binary crossentropy error."
    output = output + 1e-7
    target = target + 1e-7
    loss = - np.sum(target * np.log(output) + (1 - target) * np.log(1 - output))
    loss_gradient = output - target
    return loss, loss_gradient
