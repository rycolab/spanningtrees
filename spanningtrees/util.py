import numpy as np


def random_instance(n):
    """
    Generate a random weight matrix of size n x n
    """
    A = np.random.uniform(0, 1, size=(n, n))
    return A
