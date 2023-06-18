"""utility functions module"""

import time
import psutil
import numpy as np


def shuffle(tensor1: np.ndarray, tensor2: np.ndarray,
            batch_size: int = None) -> (np.ndarray|np.ndarray):
    """Shuffles two tensors of equal size along a axis 0 equally.

    Args:
        x: First tensors to be shuffled.
        y: Second tensors to be shuffled.
        batch_size: Number of samples to be returned [optional].
    
    Returns:
        t1_shuffled: First shuffled tensor.
        t2_shuffled: Second shuffled tensor.

    Raises:
        Error: If tensors are not of equal size along a axis 0.
    """
    if len(tensor1) != len(tensor2):
        raise Exception(f'Tensors must have equal lengths along axis 0')

    length = len(tensor1)
    shuffle_index = np.arange(length)
    batch_size = batch_size if batch_size else length
    np.random.shuffle(shuffle_index)
    t1_shuffled = tensor1[shuffle_index]
    t2_shuffled = tensor2[shuffle_index]
    return t1_shuffled[:batch_size], t2_shuffled[:batch_size]

def stopwatch(func):
    """Decorator that reports the execution time."""
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__qualname__, end-start)
        return result
    return wrap

def memlog(func):
    """Decorator that reports the current RAM usage."""
    def wrap(*args, **kwargs):
        result = func(*args, **kwargs)
        print(func.__qualname__, psutil.virtual_memory()[2], '%')
        return result
    return wrap

def set_numpy_format():
    """Sets numpy's float output to show 4 decimal places."""
    np.set_printoptions(precision=4, )
