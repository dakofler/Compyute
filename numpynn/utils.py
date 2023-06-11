# utility functions layer

import numpy as np
import time


def shuffle(x: np.ndarray, y: np.ndarray) -> (np.ndarray|np.ndarray):
    """Shuffles two arrays along a axis 0 equally.

    Args:
        x: First array to be shuffled.
        x: First array to be shuffled.
    
    Returns:
        x_shuffled: First shuffled array.
        y_shuffled: Second shuffled array.
    """
    shuffle_index = np.arange(len(x))
    np.random.shuffle(shuffle_index)
    x_shuffled = x[shuffle_index]
    y_shuffled = y[shuffle_index]
    return x_shuffled, y_shuffled

def softmax(array: np.ndarray, axis: int=1):
    """Applies the softmax function to an array along a specified axis.
    
    Args:
        array: Array to be extended.
        axis: Axis along which the softmax is to be applied.
    
    Returns:
        Array with values in the range [0, 1) who sum to 1 along the specified axis.
    """
    e = np.exp(array - np.amax(array, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def stopwatch(func):
    '''Decorator that reports the execution time.'''
  
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
          
        print(func.__qualname__, end-start)
        return result
    return wrap