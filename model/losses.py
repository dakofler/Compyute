import numpy as np

def root_mean_square(output, target):
    loss = 0.5 * np.sum(np.power(target - output, 2)) # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    loss_gradient = - (target - output)
    return loss, loss_gradient

def categorical_crossentropy(output, target): # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    loss = - np.sum(target * np.log(output))
    loss_gradient = output - target
    return loss, loss_gradient