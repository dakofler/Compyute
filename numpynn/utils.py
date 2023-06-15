# utility functions

import numpy as np
import time


def shuffle(x: np.ndarray, y: np.ndarray, batch_size: int = None) -> (np.ndarray|np.ndarray):
    """Shuffles two arrays along a axis 0 equally.

    Args:
        x: First array to be shuffled.
        y: First array to be shuffled.
        batch_size: Number of samples to be returned [optional].
    
    Returns:
        x_shuffled: First shuffled array.
        y_shuffled: Second shuffled array.
    """
    shuffle_index = np.arange(len(x))
    batch_size = batch_size if batch_size else len(x)
    np.random.shuffle(shuffle_index)
    x_shuffled = x[shuffle_index]
    y_shuffled = y[shuffle_index]
    return x_shuffled[:batch_size], y_shuffled[:batch_size]

def stopwatch(func):
    '''Decorator that reports the execution time.'''
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
          
        print(func.__qualname__, end-start)
        return result
    return wrap
