import numpy as np


# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
def root_mean_square(output, target):
    loss = 0.5 * np.sum(np.power(target - output, 2))
    loss_gradient = output - target
    return loss, loss_gradient

# https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
def categorical_crossentropy(output, target):
    output = output + 1e-7
    target = target + 1e-7
    loss = - np.sum(target * np.log(output))
    loss_gradient = output - target
    return loss, loss_gradient

# https://www.python-unleashed.com/post/derivation-of-the-binary-cross-entropy-loss-gradient
def binary_crossentropy(output, target): 
    output = output + 1e-7
    target = target + 1e-7
    loss = - np.sum(target * np.log(output) + (1 - target) * np.log(1 - output))
    loss_gradient = output - target
    return loss, loss_gradient
